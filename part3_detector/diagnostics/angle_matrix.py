"""
指纹方向夹角矩阵(机制解剖 H4,Batch 14)— 家族迁移的几何机制终审。

Batch 13 已排除表征缺失(专职探针全 ≤0.25%)与熟悉度说,楼层说只解释一部分:
机制被逼入"指纹方向的夹角几何"。本脚本对每个编码器每层:
  - 每家生成器训专职探针(该家 vs fma),取判别方向 w(换基到真-only 标准化空间,
    与 mechanism_probe 同法);
  - 单位化后输出两两余弦:ANGLE,encoder,layer,gen_i,gen_j,cos(上三角)。
本地判读(押注见 vault 共振档案 H4 节):
  ① 池→考生转移 logit-EER ~ 考生针与池成员针的平均余弦,押 ρ≤−0.5;
  ② MuQ 空间 cos(mureka, suno 族)显著低于其余编码器同格;
  ③ 夹角矩阵块结构复现家族聚类。

Usage:
  python diagnostics/angle_matrix.py --data-dir data_store/crossgen_export --encoder muq
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf


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
    for g in gens:
        idx = np.where(src == g)[0]
        if g == "suno":
            role[idx[sp[idx] == "train"]] = "train"
        else:
            perm = rng.permutation(idx)
            role[perm[:int(0.8 * len(perm))]] = "train"

    fma_tr = np.where((src == "fma") & (role == "train"))[0]

    print("ANGLE_HEADER,encoder,layer,gen_i,gen_j,cos", flush=True)
    for L in range(n_layers):
        sc = StandardScaler().fit(F[fma_tr, L])
        W = {}
        for g in gens:
            g_tr = np.where((src == g) & (role == "train"))[0]
            if len(g_tr) > args.n_train:
                g_tr = rng.choice(g_tr, args.n_train, replace=False)
            tr = np.concatenate([fma_tr, g_tr])
            clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
            w_mix = clf.named_steps["logisticregression"].coef_.ravel()
            s_mix = clf.named_steps["standardscaler"].scale_
            w = w_mix * sc.scale_ / np.maximum(s_mix, 1e-12)
            W[g] = w / np.linalg.norm(w)
        for i, gi in enumerate(gens):
            for gj in gens[i + 1:]:
                print(f"ANGLE,{args.encoder},{L},{gi},{gj},{float(W[gi] @ W[gj]):.4f}",
                      flush=True)
        print(f"  layer {L} done", flush=True)


if __name__ == "__main__":
    main()
