"""
62 手工特征批量提取(人性线 H1,2026-08-02 立项)。

背景:与教授 meeting 后拍板——黑盒分类器线之外,重启"音乐人性"的可解释路线。
本脚本把 part2 的六个特征模块(spectral/timbral/dynamics/rhythm/quantize/key)
批量跑在 crossgen_export 的 suno + fma 共同规格片段上,产出 features62.csv,
供本地 mass analysis(分布图 + 逐特征效应量 + 统计检验)。
注:lyrics_structure(歌曲结构)在 10s 片段上无意义,跳过——全长版单独立项。

Usage(服务器,纯 CPU,建议 --workers 16):
  python part2_analysis/extract_features.py \
      --data-dir data_store/crossgen_export --sources suno,fma \
      --out features62.csv --workers 16
"""
import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part2_analysis.spectral import analyze_spectral
from part2_analysis.timbral import analyze_timbral
from part2_analysis.dynamics import analyze_dynamics
from part2_analysis.rhythm import analyze_rhythm
from part2_analysis.quantize_deg import analyze_quantization
from part2_analysis.key import estimate_key

MODULES = [
    ("spec", analyze_spectral),
    ("timb", analyze_timbral),
    ("dyn", analyze_dynamics),
    ("rhy", analyze_rhythm),
    ("qnt", analyze_quantization),
]


SKIP_KEYS = {"y", "sr", "S", "S_db", "mel_db", "times", "beat_times", "onset_times",
             "onset_env", "rms", "rms_db", "chroma", "mfcc", "contrast", "grid"}


def _flatten(d, out, prefix=""):
    for k, v in d.items():
        if k in SKIP_KEYS:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, out, key + ".")
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).ravel()
            if arr.dtype.kind in "fiu" and len(arr) <= 64:   # 只收短向量(如 MFCC 统计)
                for i, x in enumerate(arr):
                    out[f"{key}_{i:02d}"] = float(x)
            # 长数组 = 画图用的时间序列/谱矩阵,丢弃
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[key] = float(v)
        elif isinstance(v, str) and len(v) < 60:
            out[key] = v
    return out


def one_clip(args):
    path, meta = args
    row = dict(meta)
    for name, fn in MODULES:
        try:
            d = fn(path)
            if "error" in d:
                row[f"{name}_error"] = d["error"]
            else:
                _flatten({k: v for k, v in d.items() if not k.startswith("_")}, row)
        except Exception as e:
            row[f"{name}_error"] = repr(e)[:80]
    try:
        best_key, best_corr, alt_key, alt_corr = estimate_key(path)
        row["best_key"] = str(best_key)
        row["best_corr"] = float(best_corr)
        row["alt_key"] = str(alt_key)
        row["alt_corr"] = float(alt_corr)
    except Exception as e:
        row["key_error"] = repr(e)[:80]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sources", default="suno,fma")
    ap.add_argument("--out", default="features62.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="烟测用:只跑前 N 条")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    want = set(args.sources.split(","))
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv"))
            if r["source"] in want]
    if args.limit:
        rows = rows[: args.limit]

    jsonl = args.out + ".jsonl"
    done_ids = set()
    old_rows = []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                r = json.loads(line)
                old_rows.append(r); done_ids.add(r["audio_id"])
            except Exception:
                pass
        print(f"resume: {len(done_ids)} already done (from {jsonl})")
    todo = [(str(data_dir / r["rel_path"]),
             {k: r.get(k, "") for k in ("audio_id", "source", "split", "label")})
            for r in rows if r["audio_id"] not in done_ids]
    print(f"{len(rows)} clips in scope, {len(todo)} to do", flush=True)
    if not todo:
        return

    results, t0 = [], time.time()
    with open(jsonl, "a") as jf, ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(one_clip, todo, chunksize=4), 1):
            results.append(row)
            jf.write(json.dumps(row) + "\n")
            if i % 50 == 0:
                jf.flush()
                print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    all_rows = old_rows + results
    fieldnames = []
    seen = set()
    for r in all_rows:
        for k in r:
            if k not in seen:
                seen.add(k); fieldnames.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    n_err = sum(1 for r in results if any(k.endswith("_error") for k in r))
    print(f"done: {len(all_rows)} rows -> {args.out} "
          f"(本轮含错误标记的行:{n_err})", flush=True)


if __name__ == "__main__":
    main()
