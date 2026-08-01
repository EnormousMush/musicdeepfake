"""
机制探针(共振机制解剖 H2+H3,Batch 13)— 每个编码器×生成器格子量两件事:

H2 针的坐标:专职探针(该生成器 vs fma)的判别方向 w,投影到 fma-train 特征的
  PCA 谱上,量"针落在第几号方差方向":
    needle_idx  = Σ w_i^2 · rank_pct_i / Σ w_i^2   (0=最大方差方向,1=最塌缩方向)
    tail_mass   = w^2 落在方差最小 25% 方向上的质量占比
  押注:编码器的"盲格"= 针系统性落在塌缩口袋(needle_idx/tail_mass 偏大)。

H3 太熟悉度:该生成器样本到 fma-train 流形(LedoitWolf 马氏)的中位距离,
  除以 fma-eval 真样本的中位距离:
    typ_ratio < 1 = 假货比真人还典型(过度典型,一阶反转风味)
  押注:MuQ 的盲格 = 在 MuQ 空间里 typ_ratio 最低(最熟悉)的生成器。

输出 CSV 行:MECH,encoder,gen,layer,probe_eer,needle_idx,tail_mass,typ_ratio
逐层输出,本地再按 oracle 层/固定层切片分析。全吃特征缓存,CPU。

Usage:
  python diagnostics/mechanism_probe.py --data-dir data_store/crossgen_export --encoder muq
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=800)
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
    gens = sorted(set(src) - {"fma"})
    print(f"Loaded {len(y)} clips, {n_layers} layers ({args.encoder})", flush=True)

    rng = np.random.default_rng(args.seed)
    role = np.array(["-"] * len(y), dtype=object)
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

    fma_tr = np.where((src == "fma") & (role == "train"))[0]
    fma_ev = np.where((src == "fma") & (role == "eval"))[0]

    print("MECH_HEADER,encoder,gen,layer,probe_eer,needle_idx,tail_mass,typ_ratio", flush=True)
    for L in range(n_layers):
        # 该层的真实流形几何:标准化 + PCA 谱 + LedoitWolf 马氏
        sc = StandardScaler().fit(F[fma_tr, L])
        Zr = sc.transform(F[fma_tr, L])
        pca = PCA().fit(Zr)
        rank_pct = np.arange(len(pca.explained_variance_)) / max(1, len(pca.explained_variance_) - 1)
        tail_cut = int(0.75 * len(rank_pct))
        lw = LedoitWolf().fit(Zr)
        mu = Zr.mean(axis=0)

        def maha(X):
            Z = sc.transform(X) - mu
            return np.einsum("ij,jk,ik->i", Z, lw.precision_, Z)

        real_med = np.median(maha(F[fma_ev, L]))

        for g in gens:
            g_tr = np.where((src == g) & (role == "train"))[0]
            if len(g_tr) > args.n_train:
                g_tr = rng.choice(g_tr, args.n_train, replace=False)
            g_ev = np.where((src == g) & (role == "eval"))[0]
            tr = np.concatenate([fma_tr, g_tr])
            clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
            ev = np.concatenate([fma_ev, g_ev])
            eer = compute_eer(y[ev], linear_clf.score(clf, F[ev, L]))["eer"]

            # 针坐标:logreg 权重先换基到"真-only 标准化"空间再投 PCA 谱
            # (pipeline 的 scaler 基是真+假混合:w_z = w_mix * s_fma / s_mix)
            w_mix = clf.named_steps["logisticregression"].coef_.ravel()
            s_mix = clf.named_steps["standardscaler"].scale_
            w = w_mix * sc.scale_ / np.maximum(s_mix, 1e-12)
            proj = pca.components_ @ w
            p2 = proj ** 2
            needle_idx = float((p2 * rank_pct).sum() / p2.sum())
            tail_mass = float(p2[tail_cut:].sum() / p2.sum())

            typ_ratio = float(np.median(maha(F[g_ev, L])) / real_med)
            print(f"MECH,{args.encoder},{g},{L},{eer*100:.2f},"
                  f"{needle_idx:.4f},{tail_mass:.4f},{typ_ratio:.4f}", flush=True)
        print(f"  layer {L} done", flush=True)


if __name__ == "__main__":
    main()
