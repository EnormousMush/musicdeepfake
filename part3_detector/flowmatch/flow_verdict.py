"""流匹配 2.0 · 零级/一级判读:对 flow_scores.csv 出预注册判决。

纯统计(阈值扫描 EER + Cohen's d),零 ML。
对比组(预注册 2026-08-19):
  零级: acestep_base vs 人类(jam+fma);acestep_turbo vs 人类
  一级: acestep_v1 / diffrhythm2 / dr1(亲戚)vs 人类,对照 musicgen / audioldm2(跨族)
  参考: suno vs 人类;拼盘(ccmixter+ianet)vs jam+fma(real 侧内部对照)

Usage:
  /Users/durunbao/Developer/frank-suno-backup/.venv/bin/python flow_verdict.py \
    "/Volumes/Seagate /honors_paper/5_results/flowmatch_v2/flow_scores_L0.csv"
"""
import csv
import sys
from collections import defaultdict

import numpy as np

SCORES = ["s1_mse_mean", "s1_cos_mean", "s2_prior_nll", "s2_fd_prior", "s2_rt_mse"]
HUMAN = ["jamendo", "fma"]
CONTRASTS = [
    ("零级 本尊", ["acestep_base"], HUMAN),
    ("零级 蒸馏子", ["acestep_turbo"], HUMAN),
    ("一级 上代本尊", ["acestep_v1"], HUMAN),
    ("一级 同族dr2", ["diffrhythm2"], HUMAN),
    ("一级 同族dr1", ["dr1"], HUMAN),
    ("一级 跨族musicgen", ["musicgen"], HUMAN),
    ("一级 跨族audioldm2", ["audioldm2"], HUMAN),
    ("二级参考 suno", ["suno"], HUMAN),
    ("real对照 拼盘", ["ccmixter", "ianet"], HUMAN),
]


def eer(pos, neg):
    """单分数阈值扫描 EER;自动取分离方向更好的一侧。"""
    pos, neg = np.asarray(pos), np.asarray(neg)
    best = 0.5
    for sign in (1, -1):
        p, n = sign * pos, sign * neg
        thr = np.unique(np.concatenate([p, n]))
        fnr = np.array([(p < t).mean() for t in thr])   # AI 判成人
        fpr = np.array([(n >= t).mean() for t in thr])  # 人判成 AI
        i = int(np.argmin(np.abs(fnr - fpr)))
        best = min(best, (fnr[i] + fpr[i]) / 2)
    return best


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def main():
    path = sys.argv[1]
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        for s in SCORES:
            if r.get(s):
                by[r["jury"]][s].append(float(r[s]))

    print("== 各团各分数均值 ==")
    juries = sorted(by)
    for s in SCORES:
        print(f"  {s:<14}" + " ".join(f"{t}={np.mean(by[t][s]):.4f}" for t in juries if by[t][s]))

    print("\n== 预注册对比(EER / Cohen's d,d>0 = AI 侧分数更高)==")
    print(f"  {'对比':<16}" + "".join(f"{s:>16}" for s in SCORES))
    for name, ai_tags, hu_tags in CONTRASTS:
        ai = {s: sum((by[t][s] for t in ai_tags if t in by), []) for s in SCORES}
        hu = {s: sum((by[t][s] for t in hu_tags if t in by), []) for s in SCORES}
        if not ai[SCORES[0]] or not hu[SCORES[0]]:
            continue
        cells = []
        for s in SCORES:
            e, d = eer(ai[s], hu[s]), cohen_d(ai[s], hu[s])
            cells.append(f"{e*100:5.1f}%/d={d:+.2f}")
        print(f"  {name:<16}" + "".join(f"{c:>16}" for c in cells))

    print("\n判据备忘:零级任一分数 |d|≥0.5 → 通过;两分数全 |d|<0.5 → 全线死(预注册)。")


if __name__ == "__main__":
    main()
