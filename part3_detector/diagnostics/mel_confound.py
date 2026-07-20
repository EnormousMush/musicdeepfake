"""
Stage 0 confound diagnostic (Option A).

Reuses the cached mel features (no audio re-decode) to test whether the
5.67% EER is driven by a high-frequency / codec-bandwidth shortcut:

  1. mean log-mel spectrum per class  -> is FMA's top end rolled off?
  2. logreg |coef| per mel band        -> which bands does the probe rely on?
  3. band ablation (low-only / high-only / cutoff sweep) -> causal test

Writes a JSON summary to data_store/diagnostics/mel_confound.json.

Usage: python diagnostics/mel_confound.py --config configs/stage0_mel_linear.yaml
"""
import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
import librosa

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import manifest as M
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def _params_key(cfg):
    blob = json.dumps({"pp": cfg["preprocess"], "enc": cfg["encoder"]}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def load_cached(cfg):
    man = M.load_manifest(cfg["paths"]["manifest"])
    cdir = Path(cfg["paths"]["cache_dir"]) / _params_key(cfg)
    X, y, sp = [], [], []
    for r in man:
        p = cdir / f"{r['audio_id']}.npy"
        if p.exists():
            X.append(np.load(p)); y.append(r["label"]); sp.append(r["split"])
    return np.vstack(X), np.array(y), np.array(sp)


def eer_on(X, y, sp, cols):
    tr, te = sp == "train", sp == "test"
    clf = linear_clf.train(X[tr][:, cols], y[tr], {"C": 1.0})
    s = linear_clf.score(clf, X[te][:, cols])
    return compute_eer(y[te], s)["eer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sr = cfg["preprocess"]["sr"]
    n_mels = cfg["encoder"]["n_mels"]

    X, y, sp = load_cached(cfg)
    print(f"Loaded cached features: X={X.shape}, AI={int((y==1).sum())}, human={int((y==0).sum())}")

    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2)  # band center freqs
    mean_cols = np.arange(n_mels)          # feature[:128]  = per-band temporal mean
    std_cols = np.arange(n_mels) + n_mels  # feature[128:]  = per-band temporal std

    # --- 1. mean log-mel spectrum per class (from the 'mean' feature block) ---
    ai_spec = X[y == 1][:, mean_cols].mean(0)
    hu_spec = X[y == 0][:, mean_cols].mean(0)
    gap = ai_spec - hu_spec  # +ve => AI louder in that band

    # --- 2. logreg standardized |coef| per band (combine mean+std parts) ---
    tr = sp == "train"
    clf = linear_clf.train(X[tr], y[tr], {"C": 1.0})
    coef = clf.named_steps["logisticregression"].coef_[0]
    band_importance = np.abs(coef[:n_mels]) + np.abs(coef[n_mels:])

    # --- 3. band ablation (causal) ---
    def cols_below(fc):
        keep = np.where(freqs <= fc)[0]
        return np.concatenate([keep, keep + n_mels])

    def cols_above(fc):
        keep = np.where(freqs >= fc)[0]
        return np.concatenate([keep, keep + n_mels])

    all_cols = np.arange(2 * n_mels)
    ablation = {
        "all_bands": eer_on(X, y, sp, all_cols),
        "low_only(<=4kHz)": eer_on(X, y, sp, cols_below(4000)),
        "low_only(<=8kHz)": eer_on(X, y, sp, cols_below(8000)),
        "high_only(>=8kHz)": eer_on(X, y, sp, cols_above(8000)),
        "high_only(>=6kHz)": eer_on(X, y, sp, cols_above(6000)),
    }
    cutoff_sweep = {f"<= {fc}Hz": eer_on(X, y, sp, cols_below(fc))
                    for fc in [1000, 2000, 4000, 6000, 8000, 11025]}

    # top bands by importance
    top = np.argsort(band_importance)[::-1][:8]
    top_bands = [{"mel_band": int(b), "center_hz": round(float(freqs[b])),
                  "importance": round(float(band_importance[b]), 3),
                  "ai_minus_human_dB": round(float(gap[b]), 2)} for b in top]

    out = {
        "n": int(len(y)),
        "ablation_eer": {k: round(v, 4) for k, v in ablation.items()},
        "cutoff_sweep_eer": {k: round(v, 4) for k, v in cutoff_sweep.items()},
        "top_importance_bands": top_bands,
        "spectrum": {  # coarse per-band, for plotting
            "center_hz": [round(float(f)) for f in freqs],
            "ai_mean_db": [round(float(v), 2) for v in ai_spec],
            "human_mean_db": [round(float(v), 2) for v in hu_spec],
        },
    }
    outp = Path("data_store/diagnostics/mel_confound.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w"), indent=2)

    # --- console report ---
    print("\n=== Ablation (test EER by band group) ===")
    for k, v in ablation.items():
        print(f"  {k:22s}: {v*100:5.2f}%")
    print("\n=== Cutoff sweep (keep only bands <= fc) ===")
    for k, v in cutoff_sweep.items():
        print(f"  {k:12s}: {v*100:5.2f}%")
    print("\n=== Top-importance mel bands ===")
    for b in top_bands:
        print(f"  band {b['mel_band']:3d} (~{b['center_hz']:5d} Hz)  "
              f"importance={b['importance']:.3f}  AI-human={b['ai_minus_human_dB']:+.1f} dB")
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
