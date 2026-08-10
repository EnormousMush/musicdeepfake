"""
H2 中窗特征批量提取:midwindow.analyze_midwindow 跑在 h2_export 的 30s 窗上。

Usage(Mac 即可,Seagate 挂载;烟测 --limit 5 先行):
  python part2_analysis/extract_midwindow.py \
      --data-dir "/Volumes/Seagate /honors_paper/3_workpacks/frank-suno-round1/h2_export" \
      --sources suno30,fma30,jam30 --out featuresH2mid.csv --workers 6
增量 jsonl 断点续跑,配方同 extract_features。
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part2_analysis.midwindow import analyze_midwindow


def one_clip(args):
    path, meta = args
    row = dict(meta)
    try:
        d = analyze_midwindow(path)
        for k, v in d.items():
            row[k] = v
    except Exception as e:
        row["mw_error"] = repr(e)[:80]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sources", default="suno30,fma30,jam30")
    ap.add_argument("--out", default="featuresH2mid.csv")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    want = set(args.sources.split(","))
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv"))
            if r["source"] in want]
    if args.limit:
        keep, seen = [], {}
        for r in rows:
            if seen.get(r["source"], 0) < args.limit:
                keep.append(r); seen[r["source"]] = seen.get(r["source"], 0) + 1
        rows = keep

    jsonl = args.out + ".jsonl"
    done, old_rows = set(), []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                r = json.loads(line)
                old_rows.append(r); done.add((r["audio_id"], r["source"]))
            except Exception:
                pass
        print(f"resume: {len(done)} done")
    todo = [(str(data_dir / r["rel_path"]),
             {k: r.get(k, "") for k in ("audio_id", "source", "split", "label")})
            for r in rows if (r["audio_id"], r["source"]) not in done]
    print(f"{len(rows)} in scope, {len(todo)} to do", flush=True)
    if todo:
        results, t0 = [], time.time()
        with open(jsonl, "a") as jf:
            if args.workers <= 1:
                it = map(one_clip, todo)
            else:
                ctx = multiprocessing.get_context("spawn")
                ex = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
                it = ex.map(one_clip, todo, chunksize=4)
            for i, row in enumerate(it, 1):
                results.append(row)
                jf.write(json.dumps(row) + "\n")
                if i % 100 == 0:
                    jf.flush()
                    print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
            if args.workers > 1:
                ex.shutdown()
        old_rows += results

    fieldnames, seen_k = [], set()
    for r in old_rows:
        for k in r:
            if k not in seen_k:
                seen_k.add(k); fieldnames.append(k)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in old_rows:
            w.writerow(r)
    print(f"done: {len(old_rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
