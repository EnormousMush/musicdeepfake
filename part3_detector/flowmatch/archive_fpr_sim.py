"""弊端三判决实验:馆藏-误伤增长曲线(2026-08-24;实验2论文方向 议题一)。

问题:档案馆判决 = 对全馆取最大匹配,档案越多,人类歌碰巧撞上某份档案的
机会越多(多重比较)。用 12 图矩阵数据模拟馆藏 1→10 的总误伤率。

数据:flow_matrix_scores.csv(E10 地图全家福,10 AI 图 × 各陪审团,n=60/团)。
设计:人类四语料(fma/jam/ccmixter/ianet)按 audio_id md5 对半分校准/评估;
每档案阈值 = 校准集 fd 的 α 分位(α=5%);判决 = 任一档案 fd < 阈值即报 AI;
馆藏顺序随机 300 次取均值;对照 Šidák 校正(α'=1-(1-α)^(1/N))。

结果(2026-08-24 开牌,存档 vault 实验2论文方向.md 议题一节):
- 未校正:5.8% → 36.5%(N=10),贴独立性理论线 40.1% → 档案对人类的误伤
  近乎独立开火,多重比较是真实硬伤;
- Šidák:N=10 压到 14.3%(样本外)/ 6.7%(全样本内界)——超出 5% 目标的
  部分是 114 首校准集撑不住 0.5% 极端分位 → 设计要求:人类校准语料要大;
- 校正的 TPR 代价集中在弱档案(dr1 63→30%,musicldm 78→42%),强档案几乎
  免费(udio 95→90%,sao 96.7→95%);
- 家族助攻:suno 自图 53% 但全馆 any 88%(亲戚档案帮着抓)。

Usage(Mac 本地,矩阵 CSV 在 Seagate 或 scratchpad):
  python part3_detector/flowmatch/archive_fpr_sim.py --csv flow_matrix_scores.csv
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

AI_MAPS = ["suno", "udio", "acestep_v1", "dr1", "diffrhythm2", "musicgen",
           "audioldm2", "musicldm", "mustango", "stable_audio_open"]
HUMANS = ["fma", "jamendo", "ccmixter", "ianet"]
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--orders", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    piv = df.pivot_table(index=["jury", "audio_id"], columns="map",
                         values="s2_fd_prior")[AI_MAPS].dropna()
    hum = piv.loc[piv.index.get_level_values("jury").isin(HUMANS)]
    h = np.array([int(hashlib.md5(a.encode()).hexdigest()[:8], 16) % 2
                  for a in hum.index.get_level_values("audio_id")])
    cal, ev = hum[h == 0], hum[h == 1]
    print(f"human clips: cal={len(cal)} eval={len(ev)}")
    tau = cal.quantile(ALPHA)

    rng = np.random.default_rng(20260824)
    orders = [rng.permutation(AI_MAPS).tolist() for _ in range(args.orders)]

    print(f"\naggregate human FPR vs archive count (per-archive alpha={ALPHA})")
    print(f"{'N':>2} {'uncorrected':>12} {'sidak':>8} {'indep-theory':>13}")
    for N in range(1, len(AI_MAPS) + 1):
        a_sidak = 1 - (1 - ALPHA) ** (1 / N)
        tau_c = cal.quantile(a_sidak)
        fu, fc = [], []
        for od in orders:
            sub = od[:N]
            fu.append((ev[sub] < tau[sub]).any(axis=1).mean())
            fc.append((ev[sub] < tau_c[sub]).any(axis=1).mean())
        print(f"{N:>2} {np.mean(fu)*100:>11.1f}% {np.mean(fc)*100:>7.1f}% "
              f"{(1-(1-ALPHA)**N)*100:>12.1f}%")

    print("\nper-archive human hit rate (eval, uncorrected):")
    hits = (ev[AI_MAPS] < tau[AI_MAPS]).mean().sort_values(ascending=False)
    print((hits * 100).round(1).to_string())

    N = len(AI_MAPS)
    a10 = 1 - (1 - ALPHA) ** (1 / N)
    tau10 = cal.quantile(a10)
    print(f"\ndetection at N={N} (own jury flagged by any archive): uncorr vs sidak")
    for g in AI_MAPS:
        gj = piv.loc[piv.index.get_level_values("jury") == g]
        if len(gj) == 0:
            continue
        to = (gj[[g]] < tau[[g]]).any(axis=1).mean()
        tu = (gj[AI_MAPS] < tau[AI_MAPS]).any(axis=1).mean()
        tc = (gj[AI_MAPS] < tau10[AI_MAPS]).any(axis=1).mean()
        print(f"  {g:<18} own-map {to*100:5.1f}% | any uncorr {tu*100:5.1f}% | any sidak {tc*100:5.1f}%")


if __name__ == "__main__":
    main()
