"""
单类检测基线(流匹配线 · 第一阶)— 只见过真实音乐的检测器能抓住什么?

科学赌注(预注册):判别式探针 + 池多样性留下了残余(Batch 7:suno 留出 17.8%,
DiffRhythm 血统 20-32%,Udio ~45%)。单类路线只用 fma-train 的特征建"真实流形",
把离流形距离当分数——若它在判别式的残余难题上反超,则"判别式学指纹 vs 单类学真实
流形"的方法论叙事成立;若不行,同样是干净的负结果。

两个密度模型(都在缓存特征上,CPU 分钟级):
  - maha: Ledoit-Wolf 收缩协方差的马氏距离(全局二阶结构)
  - knn:  k=5 近邻平均距离(局部流形结构)
选层协议与矩阵一致:在 suno-val 上选 L*(注:纯粹主义的单类选层不该看假货,
这里沿用矩阵协议保可比性,表里同时给逐生成器最优层作 oracle 参照)。

发散钩子(第 2 条纲领:每阶可外延):分数校准 / 多层集成 / PCA 维度扫描 /
真实侧换 MTG 验证"流形"是否数据集特定 / 与判别式分数融合。

Usage(服务器,特征缓存已存在):
  python diagnostics/one_class.py --data-dir data_store/crossgen_export --encoder muq
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eer import compute_eer


def fit_scorers(X_real):
    sc = StandardScaler().fit(X_real)
    Xr = sc.transform(X_real)
    lw = LedoitWolf().fit(Xr)
    P = lw.precision_
    mu = Xr.mean(axis=0)
    nn = NearestNeighbors(n_neighbors=5).fit(Xr)

    def maha(X):
        Z = sc.transform(X) - mu
        return np.einsum("ij,jk,ik->i", Z, P, Z)

    def knn(X):
        d, _ = nn.kneighbors(sc.transform(X))
        return d.mean(axis=1)

    return {"maha": maha, "knn": knn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
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
    print(f"Loaded {len(y)} clips, {n_layers} layers ({args.encoder})", flush=True)

    gens = sorted(set(src) - {"fma"})
    fma_train = (src == "fma") & (sp == "train")
    fma_test = (src == "fma") & (sp == "test")
    suno_val = (src == "suno") & (sp == "val")
    fma_val = (src == "fma") & (sp == "val")
    print(f"真实训练侧 {fma_train.sum()} | fma-test {fma_test.sum()} | 生成器({len(gens)}): {gens}")

    def pair_eer(scores_real, scores_fake):
        yy = np.concatenate([np.zeros(len(scores_real)), np.ones(len(scores_fake))])
        ss = np.concatenate([scores_real, scores_fake])
        return compute_eer(yy, ss)["eer"]

    for method in ("maha", "knn"):
        # 逐层:先给 suno(val/test)基线表,同时缓存全部生成器分数
        val_eers, results = [], {}
        for L in range(n_layers):
            scorer = fit_scorers(F[fma_train, L])[method]
            s_fma_test = scorer(F[fma_test, L])
            e_val = pair_eer(scorer(F[fma_val, L]), scorer(F[suno_val, L]))
            val_eers.append(e_val)
            results[L] = {g: pair_eer(s_fma_test, scorer(F[(src == g), L]))
                          for g in gens}
        Lstar = int(np.nanargmin(val_eers))
        print(f"\n=== 单类[{method}] 域内基线(fma-train 流形,suno vs fma)===")
        print("layer      val(suno)")
        for L in range(n_layers):
            print(f"{L:>5} {val_eers[L]*100:>10.2f}%")
        print(f"最佳层(val):L* = {Lstar}")
        print(f"\n=== 单类[{method}] 跨生成器 EER(每生成器 vs fma-test)===")
        print(f"{'generator':>20} {'@L*':>9} {'best-layer':>18}")
        for g in gens:
            bl = int(np.nanargmin([results[L][g] for L in range(n_layers)]))
            print(f"{g:>20} {results[Lstar][g]*100:>8.2f}% "
                  f"{results[bl][g]*100:>10.2f}% (L{bl})")

    print("\n判读:与判别式矩阵同测试床对表——单类若在 dr1/dr2/udio 等残余难题上"
          "反超判别式,'真实流形'路线成立;全面落后则记录为干净负结果。", flush=True)


if __name__ == "__main__":
    main()
