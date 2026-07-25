"""
Stage 0 runner — the thin vertical slice.

  build/load manifest -> preprocess -> mel encode (cached) -> linear probe -> EER

Usage:
  python run_stage0.py --config configs/stage0_mel_linear.yaml            # full subset
  python run_stage0.py --config configs/stage0_mel_linear.yaml --limit 5  # dry-run, 5/class
  python run_stage0.py --config configs/stage0_mel_linear.yaml --rebuild-manifest

Resumable: encoder features are cached to disk keyed by audio_id + params, so
re-runs and encoder/classifier swaps are cheap.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

# soundfile can't decode mp3 -> librosa falls back to audioread (works); pyloudnorm
# warns on rare clipping. Both are benign and very noisy at scale, so silence them.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # part3_detector (encoders/classifiers/eval)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root — preprocessing + manifest are Part 2
from part2_analysis import manifest as M
from part2_analysis.preprocess.audio import load_canonical
from encoders import mel as mel_encoder
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def _params_key(cfg) -> str:
    """Short hash of preprocess+encoder params -> cache namespace."""
    blob = json.dumps({"pp": cfg["preprocess"], "enc": cfg["encoder"]}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _feature(row, cfg, cache_dir, pkey):
    """Return cached or freshly computed pooled feature for one track."""
    cpath = Path(cache_dir) / pkey / f"{row['audio_id']}.npy"
    if cpath.exists():
        return np.load(cpath)
    y = load_canonical(row["path"], cfg["preprocess"])
    feat = mel_encoder.encode(y, cfg["preprocess"]["sr"], cfg["encoder"])
    cpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(cpath, feat)
    return feat


def _subsample(manifest, limit):
    """Keep the first `limit` items per class (for dry-runs)."""
    out, seen = [], {0: 0, 1: 0}
    for r in manifest:
        if seen[r["label"]] < limit:
            out.append(r)
            seen[r["label"]] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="tracks per class (dry-run)")
    ap.add_argument("--rebuild-manifest", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mpath = cfg["paths"]["manifest"]
    if args.rebuild_manifest or not os.path.exists(mpath):
        print("Building manifest...")
        man = M.build_manifest(cfg)
        M.save_manifest(man, mpath)
        print(f"  wrote {len(man)} rows -> {mpath}")
    else:
        man = M.load_manifest(mpath)
        print(f"Loaded manifest: {len(man)} rows <- {mpath}")

    if args.limit:
        man = _subsample(man, args.limit)
        print(f"Dry-run: {len(man)} rows ({args.limit}/class)")

    pkey = _params_key(cfg)
    cache_dir = cfg["paths"]["cache_dir"]
    print(f"Feature cache: {cache_dir}/{pkey}/")

    # --- extract features (resumable) ---
    X, y, splits, failures = [], [], [], []
    t0 = time.time()
    for i, row in enumerate(man, 1):
        try:
            X.append(_feature(row, cfg, cache_dir, pkey))
            y.append(row["label"])
            splits.append(row["split"])
        except Exception as e:
            failures.append((row["audio_id"], repr(e)))
        if i % 200 == 0 or i == len(man):
            print(f"  {i}/{len(man)} features ({time.time()-t0:.0f}s, {len(failures)} failed)")
    if failures:
        print(f"WARNING: {len(failures)} tracks failed to load. First few:")
        for aid, err in failures[:5]:
            print(f"    {aid}: {err}")

    X = np.vstack(X)
    y = np.array(y)
    splits = np.array(splits)

    def subset(name):
        m = splits == name
        return X[m], y[m]

    Xtr, ytr = subset("train")
    print(f"\nTrain n={len(ytr)} (AI={int(ytr.sum())}, human={int((ytr==0).sum())}), dim={X.shape[1]}")
    clf = linear_clf.train(Xtr, ytr, cfg["classifier"])

    results = {}
    for split in ("val", "test"):
        Xs, ys = subset(split)
        if len(ys) == 0 or len(set(ys)) < 2:
            print(f"{split}: skipped (n={len(ys)}, classes={set(ys.tolist())})")
            continue
        sc = linear_clf.score(clf, Xs)
        results[split] = compute_eer(ys, sc)
        r = results[split]
        print(f"{split}:  EER={r['eer']*100:.2f}%   AUC={r['auc']:.3f}   n={len(ys)}")

    # --- append one row to the results table ---
    rcsv = cfg["paths"]["results_csv"]
    Path(rcsv).parent.mkdir(parents=True, exist_ok=True)
    exists = os.path.exists(rcsv)
    with open(rcsv, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "encoder", "classifier", "params_key", "n_total",
                        "limit", "eer_val", "eer_test", "auc_test", "n_failed"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), cfg["encoder"]["name"],
                    cfg["classifier"]["name"], pkey, len(man), args.limit or "",
                    round(results.get("val", {}).get("eer", float("nan")), 4),
                    round(results.get("test", {}).get("eer", float("nan")), 4),
                    round(results.get("test", {}).get("auc", float("nan")), 4),
                    len(failures)])
    print(f"\nLogged -> {rcsv}")


if __name__ == "__main__":
    main()
