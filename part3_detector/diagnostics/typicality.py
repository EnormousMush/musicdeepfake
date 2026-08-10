"""
典型性打分(流匹配线 · 第一阶扩充,Batch 10)— 把"反转"从现象升级为信号。

第一阶(one_class.py)判决:AI 音乐不在真实流形外,而是"过度典型"——挤在众数附近,
比真人音乐自己还像人类;裸距离(单侧尺子)因此大量 EER>50% 方向反转。
本扩充仍在冻结特征空间里做(EnCodec 重建阶另归流匹配线第二阶),三个打分器:

  maha2s / knn2s  双侧典型性检验(Nalisnick 2019 药方):分数 = |距离 - 真实参照距离|,
                  "太远"和"太近"都报警。反转预言这一招能把 40-77% 直接翻正。
  tailK (64/256/1024)  小方差方向打分:PCA 按方差排序真实流形方向,只在末尾 K 个
                  小方差方向上量白化能量——检验"指纹住在距离度量忽略的方向里"。
                  (特征零中心白化,特征值下限 = 最大特征值 * 1e-6,防数值爆炸)
  lr_logo         似然比天花板(便宜版):标准化空间对角高斯,真实(fma-train)vs
                  假货池(全部生成器,留出被测者防泄漏)。它见过"假货长什么样",
                  是 real-only 打分器的公平参照上限(留出协议下)。

选层协议与矩阵/单类一致:suno-val vs fma-val 选 L*;表内同给逐生成器最优层作 oracle。
判读:双侧/tail 若把反转翻成显著 <50%,"过度典型"即为可操作信号,重建/流匹配阶值得上;
翻不动则为干净负结果(real-only 在音乐上撞墙,原因已解剖)。

Usage(服务器,特征缓存已存在,CPU 分钟级):
  python diagnostics/typicality.py --data-dir data_store/crossgen_export --encoder muq
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eer import compute_eer

TAIL_KS = (64, 256, 1024)


def fit_scorers(X_real):
    """返回 {name: scorer};全部只见 fma-train(lr_logo 除外,单独构建)。"""
    sc = StandardScaler().fit(X_real)
    Xr = sc.transform(X_real)
    scorers = {}

    # --- 双侧马氏:|d^2 - 真实参照 d^2| ---
    lw = LedoitWolf().fit(Xr)
    P = lw.precision_
    mu = Xr.mean(axis=0)

    def d2(X):
        Z = sc.transform(X) - mu
        return np.einsum("ij,jk,ik->i", Z, P, Z)

    ref_d2 = np.median(np.einsum("ij,jk,ik->i", Xr - mu, P, Xr - mu))
    scorers["maha2s"] = lambda X: np.abs(d2(X) - ref_d2)

    # --- 双侧 kNN:|knn 距离 - 真实参照距离| ---
    nn = NearestNeighbors(n_neighbors=6).fit(Xr)
    d_tr, _ = nn.kneighbors(Xr)          # 首列是自己(距离 0),丢掉
    ref_knn = np.median(d_tr[:, 1:].mean(axis=1))

    def knn2s(X):
        d, _ = nn.kneighbors(sc.transform(X), n_neighbors=5)
        return np.abs(d.mean(axis=1) - ref_knn)

    scorers["knn2s"] = knn2s

    # --- 小方差方向能量 ---
    pca = PCA().fit(Xr)
    ev = np.maximum(pca.explained_variance_, pca.explained_variance_[0] * 1e-6)
    for K in TAIL_KS:
        k = min(K, len(ev))
        V = pca.components_[-k:]
        w = 1.0 / np.sqrt(ev[-k:])

        def tail(X, V=V, w=w):
            T = (sc.transform(X) - pca.mean_) @ V.T
            return ((T * w) ** 2).mean(axis=1)

        scorers[f"tail{K}"] = tail

    return scorers, sc


def lr_scorer(sc, X_fake_pool):
    """对角高斯似然比(标准化空间):score = ll_fake - ll_real,越高越假。
    real 侧在标准化空间即 N(0, I);fake 侧对角拟合,σ 下限 1e-3。"""
    Zf = sc.transform(X_fake_pool)
    mu_f = Zf.mean(axis=0)
    sd_f = np.maximum(Zf.std(axis=0), 1e-3)

    def lr(X):
        Z = sc.transform(X)
        ll_real = -0.5 * (Z ** 2).sum(axis=1)
        ll_fake = -0.5 * (((Z - mu_f) / sd_f) ** 2).sum(axis=1) - np.log(sd_f).sum()
        return ll_fake - ll_real

    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--real-source", default="fma",
                    help="真实参照人群(fma=2010s / jamendo=2024-26);年代对照复测用 jamendo")
    args = ap.parse_args()
    R = args.real_source

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

    gens = sorted(set(src) - {"fma", "jamendo"})   # 两个真人语料都不算生成器
    fma_train = (src == R) & (sp == "train")
    fma_test = (src == R) & (sp == "test")
    suno_val = (src == "suno") & (sp == "val")
    fma_val = (src == R) & (sp == "val")
    print(f"真实参照 = {R} | 训练侧 {fma_train.sum()} | {R}-test {fma_test.sum()} | 生成器({len(gens)}): {gens}")

    def pair_eer(s_real, s_fake):
        yy = np.concatenate([np.zeros(len(s_real)), np.ones(len(s_fake))])
        ss = np.concatenate([s_real, s_fake])
        return compute_eer(yy, ss)["eer"]

    methods = ["maha2s", "knn2s"] + [f"tail{K}" for K in TAIL_KS] + ["lr_logo"]
    tables = {m: ([], {}) for m in methods}   # (val_eers, results[L][g])

    for L in range(n_layers):
        scorers, sc = fit_scorers(F[fma_train, L])
        # lr_logo:留出被测生成器的假货池(选层用留出 suno 的池)
        fake_masks = {g: (src == g) for g in gens}
        for m in methods:
            val_eers, results = tables[m]
            if m == "lr_logo":
                pool_no_suno = np.zeros(len(y), bool)
                for g in gens:
                    if g != "suno":
                        pool_no_suno |= fake_masks[g]
                s_val = lr_scorer(sc, F[pool_no_suno, L])
                e_val = pair_eer(s_val(F[fma_val, L]), s_val(F[suno_val, L]))
                res = {}
                for g in gens:
                    pool = np.zeros(len(y), bool)
                    for g2 in gens:
                        if g2 != g:
                            pool |= fake_masks[g2]
                    s = lr_scorer(sc, F[pool, L])
                    res[g] = pair_eer(s(F[fma_test, L]), s(F[fake_masks[g], L]))
                results[L] = res
            else:
                scorer = scorers[m]
                s_fma_test = scorer(F[fma_test, L])
                e_val = pair_eer(scorer(F[fma_val, L]), scorer(F[suno_val, L]))
                results[L] = {g: pair_eer(s_fma_test, scorer(F[fake_masks[g], L]))
                              for g in gens}
            val_eers.append(e_val)
        print(f"  layer {L} done", flush=True)

    for m in methods:
        val_eers, results = tables[m]
        Lstar = int(np.nanargmin(val_eers))
        print(f"\n=== 典型性[{m}] 选层(suno-val vs fma-val)===")
        print("layer      val(suno)")
        for L in range(n_layers):
            print(f"{L:>5} {val_eers[L]*100:>10.2f}%")
        print(f"最佳层(val):L* = {Lstar}")
        print(f"\n=== 典型性[{m}] 跨生成器 EER(每生成器 vs fma-test)===")
        print(f"{'generator':>20} {'@L*':>9} {'best-layer':>18}")
        for g in gens:
            bl = int(np.nanargmin([results[L][g] for L in range(n_layers)]))
            print(f"{g:>20} {results[Lstar][g]*100:>8.2f}% "
                  f"{results[bl][g]*100:>10.2f}% (L{bl})")

    print("\n判读:与 one_class.py 逐行对表——双侧/tail 把 >50% 的反转行翻到显著 <50%"
          " ⇒ '过度典型'成为可操作信号;lr_logo 列 = 见过假货长相的留出天花板,"
          "量 real-only 与它的差距。", flush=True)


if __name__ == "__main__":
    main()
