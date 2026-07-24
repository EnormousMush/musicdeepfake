"""
打乱标签 sanity check(泄漏保险丝)— 秒级,跑在已缓存特征上。

把 train 的标签随机打乱后训 probe,再在真实标签的 test 上算 EER。
✅ 期望:所有层都回到 ~50%(瞎猜)→ 没有隐性泄漏,round-1 的低 EER 是真的由数据驱动。
❌ 若某层仍明显低于 45%:存在泄漏(特征里编码了 split 结构),保存输出回来找 Claude。

Usage (server, venv active):
  python diagnostics/shuffle_check.py --data-dir data_store/subset_export_round1 --encoder mert
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    cache = data_dir / "features" / args.encoder

    F, y, sp = [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); y.append(int(r["label"])); sp.append(r["split"])
    F = np.stack(F); y = np.array(y); sp = np.array(sp)
    n_layers = F.shape[1]
    print(f"Loaded {len(y)} clips, {n_layers} layers ({args.encoder})")

    tr, te = sp == "train", sp == "test"
    rng = np.random.default_rng(args.seed)
    y_shuf = y[tr].copy()
    rng.shuffle(y_shuf)

    print(f"\n{'layer':>5} {'true test EER':>14} {'SHUFFLED test EER':>18}")
    for L in range(n_layers):
        clf_t = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
        e_true = compute_eer(y[te], linear_clf.score(clf_t, F[te, L]))["eer"]
        clf_s = linear_clf.train(F[tr, L], y_shuf, {"C": 1.0})
        e_shuf = compute_eer(y[te], linear_clf.score(clf_s, F[te, L]))["eer"]
        flag = "" if e_shuf > 0.45 else "   <-- !! 低于 45%,疑似泄漏"
        print(f"{L:>5} {e_true*100:>13.2f}% {e_shuf*100:>17.2f}%{flag}")

    print("\n✅ 判读:SHUFFLED 列应全部 ≈50%(45–55%)。是 → 无泄漏,收工。")


if __name__ == "__main__":
    main()
