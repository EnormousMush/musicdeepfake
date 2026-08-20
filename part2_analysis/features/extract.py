"""唯一批量提取 CLI(2026-08-20 大修):取代 extract_features / extract_midwindow /
extract_fulltrack 三胞胎。一个 clip 一次解码(FeatureContext 共享中间量)。

Usage:
  # 10s 线(等价旧 extract_features.py)
  python part2_analysis/features/extract.py --set h1_10s \
      --data-dir <dir含manifest.csv> --sources suno,fma --out features62.csv --workers 16
  # 30s 中窗(等价旧 extract_midwindow.py)
  python part2_analysis/features/extract.py --set h2_mid \
      --data-dir <h2_export> --sources suno30,fma30,jam30 --out featuresH2mid.csv
  # 全长(等价旧 extract_fulltrack.py;manifest 需含绝对 path 列)
  python part2_analysis/features/extract.py --set h2_full --manifest <csv> --out featuresH2full.csv
  烟测加 --limit 5(每来源前 N 条)。

与旧三胞胎的行为差异(全部是修 bug,不改数值):
  1. resume 去重键统一为 (audio_id, source)(旧 extract_features 只按 audio_id,同 id 跨来源会误跳);
  2. 列序取"注册表顺序"稳定输出(旧版靠首行运气);
  3. 事故列(ibis_* 等)不再产出(registry.SKIP_KEYS 补漏);
  4. 开跑前抽样断言输入规格(registry.SPEC),规格不符直接报错。
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from part2_analysis.features.context import FeatureContext
from part2_analysis.features.registry import SETS, SPEC, SKIP_KEYS

META_KEYS = ("audio_id", "source", "split", "label")


def _flatten(d, out, prefix=""):
    """与旧 extract_features._flatten 同构(列名规则一致),黑名单换 registry.SKIP_KEYS。"""
    for k, v in d.items():
        if k in SKIP_KEYS:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten(v, out, key + ".")
        elif isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v).ravel()
            if arr.dtype.kind in "fiu" and len(arr) <= 64:
                for i, x in enumerate(arr):
                    out[f"{key}_{i:02d}"] = float(x)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[key] = float(v)
        elif isinstance(v, str) and len(v) < 60:
            out[key] = v
    return out


def one_clip(args):
    path, meta, set_name = args
    row = dict(meta)
    ctx = FeatureContext(path)
    for tag, fn, mode in SETS[set_name]:
        try:
            d = fn(path) if mode == "path" else fn(ctx)
            if "error" in d:
                row[f"{tag}_error"] = d["error"]
            else:
                _flatten({k: v for k, v in d.items() if not k.startswith("_")}, row)
        except Exception as e:
            row[f"{tag}_error"] = repr(e)[:80]
    return row


def _check_spec(set_name, todo):
    """抽样断言输入规格 —— 数据初硬性统一从文档变成代码。"""
    import librosa
    spec = SPEC[set_name]
    for path, meta, _ in todo[:5]:
        try:
            dur = librosa.get_duration(path=path)
        except Exception as e:
            sys.exit(f"[spec-check] 无法读取 {path}: {e}")
        if spec.get("window_s"):
            if abs(dur - spec["window_s"]) > spec["tol"] * spec["window_s"]:
                sys.exit(f"[spec-check] {meta['audio_id']} 时长 {dur:.1f}s,"
                         f"与 {set_name} 期望窗 {spec['window_s']}s 不符 —— 喂错数据了?")
        elif spec.get("min_s") and dur < spec["min_s"]:
            sys.exit(f"[spec-check] {meta['audio_id']} 时长 {dur:.1f}s < {spec['min_s']}s,"
                     f"{set_name} 需要全长原盘 —— 喂错数据了?")
    print(f"[spec-check] 前 {min(5, len(todo))} 条规格通过({SPEC[set_name]['note']})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--data-dir", help="含 manifest.csv 与 rel_path 音频的目录")
    ap.add_argument("--manifest", help="独立 manifest(h2_full:需绝对 path 列)")
    ap.add_argument("--sources", default="", help="逗号分隔;空 = 全部")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="烟测:每来源前 N 条")
    args = ap.parse_args()

    if bool(args.data_dir) == bool(args.manifest):
        sys.exit("--data-dir 与 --manifest 二选一")
    man_path = Path(args.data_dir) / "manifest.csv" if args.data_dir else Path(args.manifest)
    rows = list(csv.DictReader(open(man_path)))
    if args.sources:
        want = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in want]
    if args.limit:
        keep, seen = [], {}
        for r in rows:
            if seen.get(r["source"], 0) < args.limit:
                keep.append(r); seen[r["source"]] = seen.get(r["source"], 0) + 1
        rows = keep

    def _path(r):
        return r["path"] if "path" in r and r["path"] else str(Path(args.data_dir) / r["rel_path"])

    jsonl = args.out + ".jsonl"
    done, old_rows = set(), []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                r = json.loads(line)
                old_rows.append(r); done.add((r["audio_id"], r.get("source", "")))
            except Exception:
                pass
        print(f"resume: {len(done)} done", flush=True)
    todo = [(_path(r), {k: r.get(k, "") for k in META_KEYS}, args.set)
            for r in rows if (r["audio_id"], r.get("source", "")) not in done]
    print(f"{len(rows)} in scope, {len(todo)} to do", flush=True)

    if todo:
        _check_spec(args.set, todo)
        results, t0 = [], time.time()
        with open(jsonl, "a") as jf:
            if args.workers <= 1:
                it = map(one_clip, todo)          # 串行:真实报错可见
            else:
                ctx_mp = multiprocessing.get_context("spawn")
                ex = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx_mp)
                it = ex.map(one_clip, todo, chunksize=4)
            for i, row in enumerate(it, 1):
                results.append(row)
                jf.write(json.dumps(row) + "\n")
                if i % 50 == 0:
                    jf.flush()
                    print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
            if args.workers > 1:
                ex.shutdown()
        old_rows += results

    # 稳定列序:meta 先行,特征列按全体行首次出现序(同一注册表顺序,跨批次一致)
    fieldnames, seen_k = list(META_KEYS), set(META_KEYS)
    for r in old_rows:
        for k in r:
            if k not in seen_k:
                seen_k.add(k); fieldnames.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in old_rows:
            w.writerow(r)
    n_err = sum(1 for r in old_rows if any(k.endswith("_error") for k in r))
    print(f"done: {len(old_rows)} rows -> {args.out}(含错误标记行:{n_err})", flush=True)


if __name__ == "__main__":
    main()
