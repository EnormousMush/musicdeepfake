"""
leave-one-generator-out 混训 — 生成器多样性能买到多少泛化?

把 Suno 也当作一个普通生成器,共 6 个(suno + 5 个 FakeMusicCaps)。轮流:
  留 1 个生成器完全不见 -> 用其余 5 个(+ fma)训 probe -> 在留出生成器 vs fma-test 上评 EER。
对照:单生成器训练(我们矩阵里 Suno-only 的惨状)。若混训显著改善 -> 多样性有效,
判别式仍有救;若不改善 -> 支持"跨族无共性"(师哥纲领的另一半)。

全部跑在 crossgen_export 的已缓存特征上(CPU,分钟级,无需重抽)。
FakeMusicCaps 各生成器按固定种子 80/20 分 train/eval;fma 用原 split。

Usage (server, venv active; 先跑过 run_crossgen 以生成特征缓存):
  python diagnostics/logo_mixed.py --data-dir data_store/crossgen_export --encoder mert
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

    F, y, sp, src = [], [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); y.append(int(r["label"]))
            sp.append(r["split"]); src.append(r["source"])
    F = np.stack(F); y = np.array(y); sp = np.array(sp); src = np.array(src)
    n_layers = F.shape[1]
    print(f"Loaded {len(y)} clips, {n_layers} layers ({args.encoder})")

    gens = sorted(set(src) - {"fma"})            # suno 也算一个生成器
    print(f"生成器({len(gens)}): {gens}")

    # FakeMusicCaps 生成器内部 80/20;suno 用原 split(train->train, test->eval)
    rng = np.random.default_rng(args.seed)
    role = np.array(["-"] * len(y), dtype=object)     # train / eval / -
    role[(src == "fma") & (sp == "train")] = "train"
    role[(src == "fma") & (sp == "test")] = "eval"
    for g in gens:
        idx = np.where(src == g)[0]
        if g == "suno":
            role[idx[sp[idx] == "train"]] = "train"
            role[idx[sp[idx] == "test"]] = "eval"
        else:
            perm = rng.permutation(idx)
            cut = int(0.8 * len(perm))
            role[perm[:cut]] = "train"
            role[perm[cut:]] = "eval"

    fma_eval = (src == "fma") & (role == "eval")

    def eer_of(mask, clf, L):
        if mask.sum() == 0 or len(set(y[mask].tolist())) < 2:
            return float("nan")
        return compute_eer(y[mask], linear_clf.score(clf, F[mask, L]))["eer"]

    print(f"\n=== leave-one-generator-out(训 5 个生成器 + fma,测留出者 vs fma-eval)===")
    print(f"{'held-out':>20} {'bestL':>6} {'held-out EER':>13} {'seen-gens EER':>14}")
    for held in gens:
        tr = (role == "train") & (src != held)
        held_eval = (src == held) | fma_eval          # 留出生成器全部 clips 参战
        seen_eval = (role == "eval") & (src != held)  # 见过的生成器的 eval 部分(域内参照)
        best = None
        for L in range(n_layers):
            clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
            e_seen = eer_of(seen_eval, clf, L)
            if best is None or (np.isfinite(e_seen) and e_seen < best[1]):
                best = (L, e_seen, clf)
        L, e_seen, clf = best
        e_held = eer_of(held_eval, clf, L)
        print(f"{held:>20} {L:>6} {e_held*100:>12.2f}% {e_seen*100:>13.2f}%")

    print("\n判读:held-out EER 相比单生成器训练的矩阵(Suno-only 时 31-57%)显著下降 -> "
          "多样性有效;不降 -> 支持'跨族无共性'。seen-gens 列 = 域内参照(应低)。")


if __name__ == "__main__":
    main()
