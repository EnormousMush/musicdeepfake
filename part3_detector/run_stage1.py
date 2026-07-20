"""
Stage 1 runner (server/GPU) — encoder + layer selection.

  load exported clips -> SSL encode (all layers, cached) -> per-layer linear probe -> EER

Runs on the GPU server against the folder produced by export_subset.py (rsync'd over).
Implements the testing-plan Stage 1: extract all layers once, then a cheap linear
probe per layer isolates the encoder and finds its best layer.

Usage (on the server):
  python run_stage1.py --data-dir data_store/subset_export --encoder mert
  python run_stage1.py --data-dir data_store/subset_export --encoder mert --limit 10   # dry-run
  python run_stage1.py --data-dir data_store/subset_export --encoder wav2vec2

Resumable: per-clip all-layer features cached to disk, so re-runs and probing are cheap.
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def load_manifest(data_dir):
    rows = list(csv.DictReader(open(Path(data_dir) / "manifest.csv")))
    for r in rows:
        r["label"] = int(r["label"])
    return rows


def _subsample(man, limit):
    out, seen = [], {0: 0, 1: 0}
    for r in man:
        if seen[r["label"]] < limit:
            out.append(r); seen[r["label"]] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder from export_subset.py")
    ap.add_argument("--encoder", required=True,
                    choices=["mert", "wav2vec2", "xlsr", "muq", "encodec"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--results-csv", default="data_store/results_stage1.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from encoders.ssl import SSLEncoder      # imported here so mel-only runs need no torch
    enc = SSLEncoder(args.encoder, device=args.device)
    print(f"Encoder: {args.encoder} on {enc.device} (sr={enc.sr})")

    data_dir = Path(args.data_dir)
    man = load_manifest(data_dir)
    if args.limit:
        man = _subsample(man, args.limit)
        print(f"Dry-run: {len(man)} clips ({args.limit}/class)")

    cache_dir = Path(args.cache_dir or (data_dir / "features" / args.encoder))
    cache_dir.mkdir(parents=True, exist_ok=True)

    feats, y, sp, failures = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(man, 1):
        cpath = cache_dir / f"{r['audio_id']}.npy"
        try:
            if cpath.exists():
                f = np.load(cpath)
            else:
                wav, srate = sf.read(data_dir / r["rel_path"])
                f = enc.encode_all_layers(np.asarray(wav, dtype=np.float32), srate)
                np.save(cpath, f)
            feats.append(f); y.append(r["label"]); sp.append(r["split"])
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 100 == 0 or i == len(man):
            print(f"  {i}/{len(man)} encoded ({time.time()-t0:.0f}s, {len(failures)} failed)")

    F = np.stack(feats)                 # [N, n_layers, 2D]
    y = np.array(y); sp = np.array(sp)
    n_layers = F.shape[1]
    print(f"\nFeatures: {F.shape}  (n_layers={n_layers}, dim={F.shape[2]})")

    tr, va, te = sp == "train", sp == "val", sp == "test"

    def eer_for_layer(L):
        clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
        out = {}
        for name, m in (("val", va), ("test", te)):
            if m.sum() and len(set(y[m].tolist())) == 2:
                out[name] = compute_eer(y[m], linear_clf.score(clf, F[m, L]))["eer"]
        return out

    print("\n=== Per-layer EER (linear probe) ===")
    print(f"{'layer':>5} {'val':>8} {'test':>8}")
    per_layer = []
    for L in range(n_layers):
        r = eer_for_layer(L)
        per_layer.append((L, r.get("val", float("nan")), r.get("test", float("nan"))))
        print(f"{L:>5} {r.get('val', float('nan'))*100:>7.2f}% {r.get('test', float('nan'))*100:>7.2f}%")

    valid = [p for p in per_layer if np.isfinite(p[1])]
    best = min(valid, key=lambda p: p[1]) if valid else per_layer[0]
    print(f"\nBest layer by val EER: layer {best[0]}  (val {best[1]*100:.2f}%, test {best[2]*100:.2f}%)")

    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)
    exists = os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "encoder", "n", "best_layer", "eer_val", "eer_test", "n_failed"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.encoder, len(man),
                    best[0], round(best[1], 4), round(best[2], 4), len(failures)])
    print(f"Logged -> {args.results_csv}")
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")


if __name__ == "__main__":
    main()
