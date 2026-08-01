# -*- coding: utf-8 -*-
"""共振交互分解:logit-EER 上拆 编码器主效应 + (池,考生)条件主效应 + 交互残差。
数据 = fam_{enc}_999999.txt 的跨族格(排除池内格与全池);
CI 来自 txt 里每格的 bootstrap 区间(sd ≈ 宽度/3.92),蒙特卡洛传播到交互项。
判读:Γ[编码器, 考生] < 0 = 该编码器抓该考生"好于两个主效应的预期"(共振候选格);
CI 不跨 0 才算数。
"""
import argparse
import re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=".", help="fam_{enc}_999999.txt 所在目录")
ap.add_argument("--encoders", default="mert,muq,wav2vec2",
                help="逗号分隔编码器名单,如 mert,muq,wav2vec2,xlsr,encodec")
args = ap.parse_args()
FILES = {e: f"{args.dir}/fam_{e}_999999.txt" for e in args.encoders.split(",")}
rng = np.random.default_rng(0)
EPS = 1e-3

cells = {}   # (enc, pool, gen) -> (eer, sd)
for enc, path in FILES.items():
    pool = None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"### 训练池 \[([^\]]+)\]", line)
        if m:
            pool = m.group(1)
            continue
        m = re.match(r"\s+(\S+)\s+([\d.]+)% \[\s*([\d.]+),\s*([\d.]+)\]%(\s+内)?", line)
        if m and pool and "全池" not in pool:
            g, eer, lo, hi = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            if m.group(5):           # 池内格,剔除
                continue
            cells[(enc, pool, g)] = (eer / 100, (hi - lo) / 100 / 3.92)

encs = sorted({k[0] for k in cells})
conds = sorted({(k[1], k[2]) for k in cells})
gens = sorted({k[2] for k in cells})
print(f"跨族格:{len(cells)}(= {len(encs)} 编码器 × {len(conds)} 条件)")

def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))

def decompose(draw):
    Y = np.array([[draw[(e, c[0], c[1])] for c in conds] for e in encs])   # [E, C]
    L = logit(Y)
    mu = L.mean()
    a = L.mean(axis=1) - mu                  # 编码器主效应
    b = L.mean(axis=0) - mu                  # 条件主效应
    R = L - mu - a[:, None] - b[None, :]     # 交互残差
    G = np.zeros((len(encs), len(gens)))     # 按考生聚合
    for j, g in enumerate(gens):
        idx = [i for i, c in enumerate(conds) if c[1] == g]
        G[:, j] = R[:, idx].mean(axis=1)
    var_main = np.var(a[:, None] + b[None, :])
    return G, np.var(R) / (np.var(R) + var_main), a

point = {k: v[0] for k, v in cells.items()}
G0, share0, a0 = decompose(point)

N = 3000
Gs = np.zeros((N, len(encs), len(gens)))
shares = np.zeros(N)
for t in range(N):
    draw = {k: float(np.clip(rng.normal(v[0], v[1]), EPS, 1 - EPS)) for k, v in cells.items()}
    Gs[t], shares[t], _ = decompose(draw)

lo, hi = np.percentile(Gs, [2.5, 97.5], axis=0)

print(f"\n主效应(logit 尺度,负=整体更强): " +
      ", ".join(f"{e}={a0[i]:+.2f}" for i, e in enumerate(encs)))
print(f"交互占跨族格方差比例:{share0*100:.1f}%  [{np.percentile(shares,2.5)*100:.1f}, {np.percentile(shares,97.5)*100:.1f}]%")
print("\n交互矩阵 Γ[编码器, 考生](负=好于预期;* = 95%CI 不跨 0)")
print(f"{'考生':>18} " + "".join(f"{e:>22}" for e in encs))
for j, g in enumerate(gens):
    row = ""
    for i in range(len(encs)):
        star = "*" if (lo[i, j] > 0 or hi[i, j] < 0) else " "
        row += f"{G0[i,j]:+.2f} [{lo[i,j]:+.2f},{hi[i,j]:+.2f}]{star}"
    print(f"{g:>18} " + row)

print("\n预注册/线索格核对:")
for enc, g, tag in [("muq", "audioldm2", "预测格2 MuQ×lofi"), ("muq", "musicldm", "预测格2"),
                    ("muq", "mustango", "预测格2"), ("mert", "MusicGen_medium", "预测格1 MERT×MusicGen"),
                    ("mert", "mureka", "线索格 MERT×Mureka")]:
    i, j = encs.index(enc), gens.index(g)
    star = "显著" if (lo[i, j] > 0 or hi[i, j] < 0) else "不显著"
    print(f"  {tag:>24}: Γ={G0[i,j]:+.2f} [{lo[i,j]:+.2f},{hi[i,j]:+.2f}] {star}")
