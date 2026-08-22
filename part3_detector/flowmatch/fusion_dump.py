"""融合检验前置(2026-08-22):把判别式探针的逐 clip 分数补 dump 到 test/heldout。

复刻 era_holdout.py 的 mixed 配方(训练 suno-train vs fma-train+jam-train-inst,
混合 val 选层)——协议一字不改,只是把 test split 的逐 clip 分数也写出来
(Batch I 当年只 dump 了 heldout 815)。

Usage(服务器 .venv 旧环境,和 Batch I 同一套;tmux):
  python fusion_dump.py --data-dir data_store/crossgen_export \
    --heldout-dir data_store/heldout_export --encoder muq --inst-only
输出:fusion_probe_scores_<enc>.csv(audio_id, source, split, score_mixed)
"""
import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import time

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "diagnostics"))
from classifiers import linear as linear_clf
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

SOURCES = ("suno", "fma", "jamendo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--heldout-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--inst-only", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache = data_dir / "features" / args.encoder
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv"))
            if r["source"] in SOURCES]
    if args.inst_only:
        ids = instrumental_ids()
        if ids is not None:
            rows = filter_rows(rows, ids)
    F, meta = [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); meta.append(r)
    F = np.stack(F)
    sp = np.array([m["split"] for m in meta])
    src = np.array([m["source"] for m in meta])
    print(f"crossgen: {len(meta)} | " + " ".join(f"{s}:{(src==s).sum()}" for s in SOURCES), flush=True)

    hdir = Path(args.heldout_dir)
    hcache = hdir / "features" / args.encoder
    hrows = list(csv.DictReader(open(hdir / "manifest.csv")))
    H, hmeta = [], []
    for r in hrows:
        p = hcache / f"{r['audio_id']}.npy"
        if p.exists():
            H.append(np.load(p)); hmeta.append(r)
    H = np.stack(H)
    print(f"heldout: {len(hmeta)}", flush=True)

    y = (src == "suno").astype(int)
    m_va = (sp == "val")
    tr = (sp == "train")                                   # mixed 配方:全部三源的 train
    n_layers = F.shape[1]
    t0 = time.time()
    best = None
    for L in range(n_layers):
        clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
        ev = compute_eer(y[m_va], linear_clf.score(clf, F[m_va, L]))["eer"]
        if best is None or ev < best[1]:
            best = (L, ev, clf)
    L, ev, clf = best
    print(f"mixed L*={L} (val EER {ev*100:.2f}%, {time.time()-t0:.0f}s)", flush=True)

    m_te = (sp == "test")
    s_te = linear_clf.score(clf, F[m_te, L])
    s_ho = linear_clf.score(clf, H[:, L])

    out = f"fusion_probe_scores_{args.encoder}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["audio_id", "source", "split", "score_mixed"])
        for m, s in zip(np.array(meta, dtype=object)[m_te], s_te):
            w.writerow([m["audio_id"], m["source"], "test", f"{float(s):.6f}"])
        for m, s in zip(hmeta, s_ho):
            w.writerow([m["audio_id"], m["source"], "heldout", f"{float(s):.6f}"])
    # 基线 EER 备查(test 上 suno vs 各人类源)
    yy = y[m_te]
    for tag in ("fma", "jamendo"):
        mm = np.isin(src[m_te], ["suno", tag])
        e = compute_eer(yy[mm], s_te[mm])["eer"]
        print(f"probe-alone vs {tag}: EER {e*100:.2f}%", flush=True)
    print(f"DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
