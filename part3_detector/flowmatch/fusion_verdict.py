"""融合裁决(2026-08-22 预注册):过程分数有没有判别式之外的正交增益?

对齐(audio_id)两张表:
  A = fusion_probe_scores_<enc>.csv(判别式 mixed 探针逐 clip 分数)
  B = flow_b_suno_scores.csv(路 B 过程分数 5 项)
5 折分层交叉验证,同折同 clip 比三个模型:
  M1 只用探针分数(基线);M2 只用过程 5 分;M3 探针+过程 6 维。
判据(预注册):M3 的 CV-EER 显著低于 M1(配对差 > 1pp 且方向一致)=> 正交增益成立。

Usage(Mac):
  .venv/bin/python fusion_verdict.py <probe_csv> <flow_csv>
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROC = ["s1_mse_mean", "s1_cos_mean", "s2_prior_nll", "s2_fd_prior", "s2_rt_mse"]


def eer(y, s):
    y, s = np.asarray(y), np.asarray(s)
    thr = np.unique(s)
    fnr = np.array([(s[y == 1] < t).mean() for t in thr])
    fpr = np.array([(s[y == 0] >= t).mean() for t in thr])
    i = int(np.argmin(np.abs(fnr - fpr)))
    return (fnr[i] + fpr[i]) / 2


def main():
    probe_csv, flow_csv = sys.argv[1], sys.argv[2]
    probe = {r["audio_id"]: r for r in csv.DictReader(open(probe_csv))}
    rows = []
    for r in csv.DictReader(open(flow_csv)):
        p = probe.get(r["audio_id"])
        if p is None:
            continue
        rows.append(dict(audio_id=r["audio_id"], jury=r["jury"],
                         y=1 if r["jury"] == "suno_test" else 0,
                         probe=float(p["score_mixed"]),
                         **{k: float(r[k]) for k in PROC}))
    y = np.array([r["y"] for r in rows])
    print(f"joined {len(rows)} clips (suno {y.sum()}, human {(y==0).sum()}) | "
          + " ".join(f"{j}:{sum(1 for r in rows if r['jury']==j)}"
                     for j in sorted({r['jury'] for r in rows})))

    Xp = np.array([[r["probe"]] for r in rows])
    Xf = np.array([[r[k] for k in PROC] for r in rows])
    X3 = np.hstack([Xp, Xf])
    models = {"M1 探针 alone": Xp, "M2 过程 alone": Xf, "M3 探针+过程": X3}

    skf = StratifiedKFold(5, shuffle=True, random_state=20260822)
    fold_eers = defaultdict(list)
    for tr, te in skf.split(Xp, y):
        for name, X in models.items():
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            s = clf.predict_proba(sc.transform(X[te]))[:, 1]
            fold_eers[name].append(eer(y[te], s))

    print("\n== 5 折 CV-EER ==")
    for name in models:
        es = np.array(fold_eers[name])
        print(f"  {name:<14} {es.mean()*100:5.2f}% ± {es.std()*100:.2f}  折明细 "
              + " ".join(f"{e*100:.1f}" for e in es))
    d = np.array(fold_eers["M1 探针 alone"]) - np.array(fold_eers["M3 探针+过程"])
    print(f"\n配对差(M1−M3):{d.mean()*100:+.2f}pp,同向折数 {np.sum(d>0)}/5")
    print("判据:均值差 >1pp 且 ≥4/5 折同向 => 正交增益成立")


if __name__ == "__main__":
    main()
