"""入档演习本地分析(2026-08-26):主表/遗忘/样本效率/未知桶。"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

GENS = ["MusicGen_medium", "audioldm2", "musicldm", "mustango",
        "stable_audio_open", "inspire", "dr1", "diffrhythm2",
        "acestep", "levo", "mureka", "suno"]

nd = pd.read_csv("drill_needle_logits.csv")
bs = pd.read_csv("drill_baseline_scores.csv")
cal_n = nd[nd.role == "cal"]
evh_n = nd[nd.role == "eval_human"]
eva_n = nd[nd.role == "eval_ai"]


def needle_step(t, alpha, budget=300):
    enrolled = GENS[:t]
    a_s = 1 - (1 - alpha) ** (1 / t)
    cols = [f"{g}__b{budget}" for g in enrolled]
    tau = {c: cal_n[c].quantile(1 - a_s) for c in cols}
    fire_h = np.column_stack([evh_n[c] > tau[c] for c in cols]).any(axis=1)
    fpr = fire_h.mean()
    tprs, attr_ok, attr_n = {}, 0, 0
    for g in GENS:
        sub = eva_n[eva_n.source == g]
        fire = np.column_stack([sub[c] > tau[c] for c in cols])
        tprs[g] = fire.any(axis=1).mean()
        if g in enrolled:
            hit = fire.any(axis=1)
            if hit.sum():
                z = np.column_stack([(sub[c] - cal_n[c].mean()) / cal_n[c].std() for c in cols])
                pred = np.array(enrolled)[z.argmax(axis=1)]
                attr_ok += (pred[hit] == g).sum(); attr_n += hit.sum()
    return fpr, tprs, (attr_ok / attr_n if attr_n else np.nan)


print("===== A. 针层主表(budget=300;每步 Šidák 全馆误报预算) =====")
for alpha in [0.05, 0.01]:
    print(f"\n-- 总误报预算 {alpha*100:.0f}% --")
    print(f"{'t':>2} {'newly-enrolled':<18} {'FPR':>6} {'meanTPR(in)':>12} {'TPR(new)':>9} {'attr-acc':>9}")
    for t in range(1, 13):
        fpr, tprs, acc = needle_step(t, alpha)
        ins = np.mean([tprs[g] for g in GENS[:t]])
        print(f"{t:>2} {GENS[t-1]:<18} {fpr*100:>5.1f}% {ins*100:>11.1f}% "
              f"{tprs[GENS[t-1]]*100:>8.1f}% {acc*100:>8.1f}%")

print("\n===== B. 样本效率:全馆 t=12,Šidák 5%,各预算 per-gen TPR =====")
print(f"{'gen':<18}", *[f"b{b:>5}" for b in [100, 300, 1000]])
for g in GENS:
    row = []
    for b in [100, 300, 1000]:
        fpr, tprs, _ = needle_step(12, 0.05, budget=b)
        row.append(f"{tprs[g]*100:5.1f}%")
    print(f"{g:<18}", *row)

cal_b = bs[bs.role == "cal"]; evh_b = bs[bs.role == "eval_human"]; eva_b = bs[bs.role == "eval_ai"]
print("\n===== C. 对照组(budget=300,单头,阈值=cal 5% FPR)vs 针层 =====")
print("meanTPR(enrolled) 轨迹:")
hdr = f"{'t':>2} {'new':<18}" + "".join(f"{m:>9}" for m in ["retrain", "finetune", "replay", "ewc", "noupdate", "NEEDLE"])
print(hdr)
for t in range(1, 13):
    row = f"{t:>2} {GENS[t-1]:<18}"
    for m in ["retrain", "finetune", "replay", "ewc", "noupdate"]:
        c = f"{m}__t{t:02d}"
        tau = cal_b[c].quantile(0.95)
        tpr = np.mean([ (eva_b[eva_b.source == g][c] > tau).mean() for g in GENS[:t] ])
        row += f"{tpr*100:>8.1f}%"
    _, tprs, _ = needle_step(t, 0.05)
    row += f"{np.mean([tprs[g] for g in GENS[:t]])*100:>8.1f}%"
    print(row)

print("\n遗忘:第一家(MusicGen)的 TPR 轨迹(入档后随步数):")
row = {m: [] for m in ["retrain", "finetune", "replay", "ewc", "noupdate"]}
for t in range(1, 13):
    for m in row:
        c = f"{m}__t{t:02d}"
        tau = cal_b[c].quantile(0.95)
        row[m].append((eva_b[eva_b.source == "MusicGen_medium"][c] > tau).mean() * 100)
for m, v in row.items():
    print(f"  {m:<9}", " ".join(f"{x:5.1f}" for x in v))
fpr, tprs, _ = needle_step(12, 0.05)
print(f"  NEEDLE    结构性不变: {tprs['MusicGen_medium']*100:.1f}%(任一步都一样)")

print("\n===== D. 未知桶:未入档生成器的歌成不成簇(KMeans, ARI) =====")
feats = np.load("drill_evalai_feats.npy")
meta = pd.read_csv("drill_evalai_meta.csv")
Xp = feats.mean(axis=1) if feats.ndim == 3 else feats
for t in [3, 6, 9]:
    unk = [g for g in GENS[t:]]
    m = meta.source.isin(unk).values
    Xu = Xp[m]; yu = meta.source[m].values
    km = KMeans(n_clusters=len(unk), n_init=10, random_state=0).fit(Xu)
    print(f"  t={t:>2} 未入档 {len(unk)} 家: ARI = {adjusted_rand_score(yu, km.labels_):.3f} (n={len(yu)})")
