# -*- coding: utf-8 -*-
"""Batch 14 夹角说终审(2026-08-12 重建版:原 angle_verdict.py 未入库,按 vault 记录重写)。

输入(均在 Seagate honors paper 原始数据/):
  angle_{enc}.txt  —— angle_matrix.py 产物,每层 66 对生成器指纹方向余弦
  fam13_{enc}.txt  —— family_matrix.py --dump-layers 产物,每格逐层转移 EER + 池成员 + L*

三个开牌 + 一张附赠表,全部只用 numpy(Spearman 自己用秩相关算,不依赖 scipy):
  ① 主注:跨族格里"考生针 vs 池成员针的平均余弦" vs logit-EER 的 Spearman ρ(押 ≤ -0.5)
  ② 关键格:cos(mureka, suno 族三员均值) @ 各家 suno_fam 池选层,含"该层全对基线"与"超额对齐"
  ③ 家族块结构:逐层平均的族内 vs 跨族箭头余弦
  附赠:每个编码器里 mureka 的最近邻箭头(逐层平均余弦)
"""
import argparse, re, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=".")
ap.add_argument("--encoders", default="mert,muq,wav2vec2,xlsr,encodec")
args = ap.parse_args()
ENCS = args.encoders.split(",")
EPS = 1e-3

def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))

def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float); order = v.argsort()
        r = np.empty(len(v), float); r[order] = np.arange(len(v), dtype=float)
        # 处理并列:同值取平均秩
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")

def load_angles(path):
    """-> {layer: {frozenset({gi,gj}): cos}}"""
    A = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        if not line.startswith("ANGLE,"): continue
        p = line.rstrip("\n").split(",")
        if p[1] == "encoder": continue
        A[int(p[2])][frozenset((p[3], p[4]))] = float(p[5])
    return A

def load_fam13(path):
    """-> (pools: {pool: {'members':[...], 'Lstar':int}}, prof: {(pool,gen): [per-layer EER]})"""
    pools, prof = {}, {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"### 训练池 \[([^\](]+)[^\]]*\] 成员=\[([^\]]*)\].*?L\*=(\d+)", line)
        if m:
            if "全池" in m.group(1): continue
            pools[m.group(1).strip()] = {
                "members": re.findall(r"'([^']+)'", m.group(2)),
                "Lstar": int(m.group(3))}
            continue
        if line.startswith("LAYER_PROFILE,"):
            p = line.rstrip("\n").split(",")
            pool = p[1].split("(")[0].strip()
            if "全池" in pool: continue
            prof[(pool, p[2])] = [float(x) for x in p[8:]]
    return pools, prof

def cos(A, layer, a, b):
    if a == b: return 1.0
    return A[layer].get(frozenset((a, b)))

print("=" * 78)
print("Batch 14 夹角说终审 · 重建复算(仅 numpy)")
print("=" * 78)

rows1, rows2, rows3, rows4 = [], [], [], []
for enc in ENCS:
    A = load_angles(os.path.join(args.dir, f"angle_{enc}.txt"))
    pools, prof = load_fam13(os.path.join(args.dir, f"fam13_{enc}.txt"))
    layers = sorted(A)
    gens = sorted({g for L in A.values() for k in L for g in k})
    fam = {g: p for p, d in pools.items() for g in d["members"]}

    # ① 主注
    xs, ys = [], []
    for pool, d in pools.items():
        Ls, mem = d["Lstar"], d["members"]
        for g in gens:
            if g in mem: continue
            c = [cos(A, Ls, g, m) for m in mem]
            if any(v is None for v in c): continue
            if (pool, g) not in prof: continue
            xs.append(float(np.mean(c))); ys.append(prof[(pool, g)][Ls] / 100)
    rho = spearman(xs, ys := logit(ys))
    rows1.append((enc, len(xs), rho))

    # ② 关键格
    sf = pools.get("suno_fam")
    if sf:
        Ls = sf["Lstar"]
        cm = float(np.mean([cos(A, Ls, "mureka", m) for m in sf["members"]]))
        base = float(np.mean(list(A[Ls].values())))
        rows2.append((enc, Ls, cm, base, cm - base))

    # ③ 块结构 + 附赠最近邻
    wi, cr = [], []
    for L in layers:
        for k, v in A[L].items():
            a, b = tuple(k)
            if a in fam and b in fam:
                (wi if fam[a] == fam[b] else cr).append(v)
    rows3.append((enc, float(np.mean(wi)), float(np.mean(cr)), float(np.mean(wi)) - float(np.mean(cr))))

    nb = {g: float(np.mean([A[L][frozenset(("mureka", g))] for L in layers
                            if frozenset(("mureka", g)) in A[L]]))
          for g in gens if g != "mureka"}
    top = sorted(nb.items(), key=lambda kv: -kv[1])[:3]
    rows4.append((enc, top))

print("\n【开牌①·主注】跨族格:考生针 vs 池成员针平均余弦  ~  logit-EER(押 Spearman ρ ≤ -0.5)")
print(f"{'编码器':>10}{'跨族格数':>10}{'Spearman ρ':>14}   判读")
for enc, n, rho in rows1:
    print(f"{enc:>10}{n:>10}{rho:>14.2f}   {'命中' if rho <= -0.5 else '未达'}")

print("\n【开牌②·关键格】cos(mureka, suno族均值) @ 各家 suno_fam 选层")
print(f"{'编码器':>10}{'L*':>5}{'绝对值':>10}{'该层全对基线':>14}{'超额对齐':>12}")
for enc, Ls, cm, base, ex in rows2:
    print(f"{enc:>10}{Ls:>5}{cm:>10.3f}{base:>14.3f}{ex:>+12.3f}")

print("\n【开牌③·家族块结构】逐层平均箭头余弦")
print(f"{'编码器':>10}{'族内':>10}{'跨族':>10}{'分离度':>10}")
for enc, w, c, s in rows3:
    print(f"{enc:>10}{w:>10.3f}{c:>10.3f}{s:>+10.2f}")

print("\n【附赠·考古几何盖章】mureka 的最近邻箭头(逐层平均余弦,前三)")
for enc, top in rows4:
    print(f"{enc:>10}  " + " | ".join(f"{g} {v:.2f}" for g, v in top))
print()
