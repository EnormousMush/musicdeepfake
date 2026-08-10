"""
H2b 全长结构特征批量提取(fulltrack.analyze_fulltrack)。

manifest 需含列:audio_id, source, split, path(绝对路径,原盘 mp3)。
Usage(Mac,烟测 --limit 3 先行):
  python part2_analysis/extract_fulltrack.py --manifest <csv> --out featuresH2full.csv --workers 4
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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part2_analysis.fulltrack import analyze_fulltrack


def one_clip(args):
    path, meta = args
    row = dict(meta)
    try:
        row.update(analyze_fulltrack(path))
    except Exception as e:
        row["ft_error"] = repr(e)[:80]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="featuresH2full.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
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
                old_rows.append(r); done.add(r["audio_id"])
            except Exception:
                pass
        print(f"resume: {len(done)} done")
    todo = [(r["path"], {k: r.get(k, "") for k in ("audio_id", "source", "split")})
            for r in rows if r["audio_id"] not in done]
    print(f"{len(rows)} in scope, {len(todo)} to do", flush=True)
    if todo:
        results, t0 = [], time.time()
        with open(jsonl, "a") as jf:
            if args.workers <= 1:
                it = map(one_clip, todo)
            else:
                ctx = multiprocessing.get_context("spawn")
                ex = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
                it = ex.map(one_clip, todo, chunksize=2)
            for i, row in enumerate(it, 1):
                results.append(row)
                jf.write(json.dumps(row) + "\n")
                if i % 50 == 0:
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
