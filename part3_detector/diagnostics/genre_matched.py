"""
Genre-matched control (Stage 4, cheap) — quantify how much of MERT's ~1.3% EER is a
genre-distribution confound, using the ALREADY-CACHED features (instant, no new data).

FMA-instrumental skews classical/ambient/electronic; Suno is 8 balanced pop genres. If the
low EER is just "different genres," restricting BOTH sides to shared genres should make the
task much harder (EER rises). If EER stays low, genre isn't the (only) confound.

Usage (server, in detector dir, venv active):
  python diagnostics/genre_matched.py --data-dir data_store/subset_export --encoder mert --layer 6
"""
import argparse
import ast
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer

SUNO_GENRES = {"blues", "classical", "country", "electronic", "hiphop", "jazz", "pop", "rock"}


def clip_genres(row):
    """Return the set of lowercased genre names for a clip."""
    if int(row["label"]) == 1:                      # Suno: clean single genre
        return {row["genre"].strip().lower()}
    try:                                            # FMA: a JSON-ish list of dicts
        items = ast.literal_eval(row["genre"])
        return {d["genre_title"].strip().lower() for d in items}
    except Exception:
        return set()


def eer_on(mask, X, y, sp):
    tr = mask & (sp == "train")
    te = mask & (sp == "test")
    if tr.sum() < 20 or te.sum() < 10 or len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
        return None, int(tr.sum()), int(te.sum())
    clf = linear_clf.train(X[tr], y[tr], {"C": 1.0})
    eer = compute_eer(y[te], linear_clf.score(clf, X[te]))["eer"]
    return eer, int(tr.sum()), int(te.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--layer", type=int, default=6)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    cache = data_dir / "features" / args.encoder

    X, y, sp, gsets = [], [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if not p.exists():
            continue
        X.append(np.load(p)[args.layer])
        y.append(int(r["label"])); sp.append(r["split"]); gsets.append(clip_genres(r))
    X = np.vstack(X); y = np.array(y); sp = np.array(sp)
    print(f"Loaded {len(y)} clips at layer {args.layer} (AI={int((y==1).sum())}, human={int((y==0).sum())})")

    # which Suno genres also appear in FMA, with enough samples on both sides?
    shared = []
    for g in sorted(SUNO_GENRES):
        fma_n = sum((int(y[i]) == 0) and (g in gsets[i]) for i in range(len(y)))
        suno_n = sum((int(y[i]) == 1) and (g in gsets[i]) for i in range(len(y)))
        if fma_n >= 20 and suno_n >= 20:
            shared.append(g)
        print(f"  genre '{g:10s}': FMA={fma_n:4d}  Suno={suno_n:4d}"
              + ("   <- shared" if g in shared else ""))

    all_mask = np.ones(len(y), dtype=bool)
    e, ntr, nte = eer_on(all_mask, X, y, sp)
    print(f"\nALL genres (baseline):        EER={e*100:.2f}%  (train {ntr}, test {nte})")

    if shared:
        matched = np.array([len(gsets[i] & set(shared)) > 0 for i in range(len(y))])
        e, ntr, nte = eer_on(matched, X, y, sp)
        tag = "+".join(shared)
        msg = f"EER={e*100:.2f}%" if e is not None else "n/a (too few)"
        print(f"GENRE-MATCHED ({tag}): {msg}  (train {ntr}, test {nte})")

        for g in shared:
            m = np.array([g in gsets[i] for i in range(len(y))])
            e, ntr, nte = eer_on(m, X, y, sp)
            msg = f"EER={e*100:.2f}%" if e is not None else "n/a (too few)"
            print(f"   only '{g}': {msg}  (train {ntr}, test {nte})")
    else:
        print("No shared genres with enough samples on both sides.")


if __name__ == "__main__":
    main()
