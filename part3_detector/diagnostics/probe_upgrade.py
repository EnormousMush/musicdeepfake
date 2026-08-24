"""强分类器复现实验(2026-08-24 预注册;实验2论文方向 Q3)。

目的:验证线性探针上观察到的**结构**在更强分类器下复现——不是刷分。
预注册赌注(Müller 2024 + 我们的机制证据):域内 EER 略降;LOGO 留出与
real 侧最坏情况的**结构不变**(该塌的照塌,排序不变)。

设计:同一批缓存 SSL 特征(crossgen bed),五种头 × 三个协议:
  头:linear(基线复现)/ mlp1(512)/ mlp2(512-256)/ fuse_lin(全层拼接+线性)
     / fuse_mlp(全层拼接+512)
  协议:P1 mixed 域内(suno vs fma+jam-inst,混合 val 选层)
       P2 real 侧最坏情况(fma-only 训练 → jamendo 测,era 线的 18× 退化案)
       P3 LOGO(留一生成器:其余生成器+suno vs fma 训 → 留出者 vs fma-test)
单层头逐层扫描按 val 选层(与历史口径一致);融合头免选层。

Usage(服务器 .venv-flow2,tmux):
  python diagnostics/probe_upgrade.py --data-dir data_store/crossgen_export \
    --encoder muq --inst-only
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENS = ["MusicGen_medium", "audioldm2", "musicldm", "mustango", "stable_audio_open"]


def load_bed(args):
    data_dir = Path(args.data_dir)
    cache = data_dir / "features" / args.encoder
    keep_src = set(["suno", "fma", "jamendo"] + GENS)
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv")) if r["source"] in keep_src]
    if args.inst_only:
        ids = instrumental_ids()
        if ids is not None:
            rows = filter_rows(rows, ids)
    F, meta = [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); meta.append(r)
    F = np.stack(F).astype(np.float32)
    sp = np.array([m["split"] for m in meta])
    src = np.array([m["source"] for m in meta])
    print(f"bed: {F.shape} | " + " ".join(f"{s}:{(src==s).sum()}" for s in sorted(set(src))), flush=True)
    return F, sp, src


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_head(Xtr, ytr, Xva, yva, hidden, seed=20260824, epochs=40):
    """hidden=[] 即线性(带权重衰减的逻辑回归,torch 版);返回打分函数。"""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.from_numpy((Xtr - mu) / sd).to(DEVICE)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(DEVICE)
    Xva_t = torch.from_numpy((Xva - mu) / sd).to(DEVICE)
    net = MLP(Xtr.shape[1], hidden).to(DEVICE)
    pos_w = torch.tensor([(ytr == 0).sum() / max(1, (ytr == 1).sum())], device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xtr_t)
    best = (None, 1.0)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sva = torch.sigmoid(net(Xva_t)).cpu().numpy()
        ev = compute_eer(yva, sva)["eer"]
        if ev < best[1]:
            best = ({k: v.clone() for k, v in net.state_dict().items()}, ev)
    net.load_state_dict(best[0])
    net.eval()

    def score(X):
        with torch.no_grad():
            Xt = torch.from_numpy((X - mu) / sd).to(DEVICE)
            return torch.sigmoid(net(Xt)).cpu().numpy()
    return score, best[1]


HEADS = {
    "linear":   dict(hidden=[], fuse=False),
    "mlp1":     dict(hidden=[512], fuse=False),
    "mlp2":     dict(hidden=[512, 256], fuse=False),
    "fuse_lin": dict(hidden=[], fuse=True),
    "fuse_mlp": dict(hidden=[512], fuse=True),
}


def run_protocol(F, sp, src, tr_mask, va_mask, tests, head):
    """tests: dict name -> (mask, y);单层头逐层 val 选层,融合头拼接全层。"""
    cfg = HEADS[head]
    y = (np.isin(src, ["fma", "jamendo"]) == False).astype(int)  # 1 = 生成器侧
    if cfg["fuse"]:
        Xall = F.reshape(F.shape[0], -1)
        score, ev = fit_head(Xall[tr_mask], y[tr_mask], Xall[va_mask], y[va_mask], cfg["hidden"])
        out = {name: compute_eer(yy, score(Xall[m]))["eer"] for name, (m, yy) in tests.items()}
        return out, -1, ev
    best = None
    for L in range(F.shape[1]):
        score, ev = fit_head(F[tr_mask, L], y[tr_mask], F[va_mask, L], y[va_mask], cfg["hidden"])
        if best is None or ev < best[2]:
            best = (score, L, ev)
    score, L, ev = best
    out = {name: compute_eer(yy, score(F[m, L]))["eer"] for name, (m, yy) in tests.items()}
    return out, L, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--inst-only", action="store_true")
    ap.add_argument("--heads", default="linear,mlp1,mlp2,fuse_lin,fuse_mlp")
    args = ap.parse_args()
    F, sp, src = load_bed(args)
    y_gen = (np.isin(src, ["fma", "jamendo"]) == False).astype(int)

    for head in args.heads.split(","):
        t0 = time.time()
        print(f"\n===== head = {head} =====", flush=True)

        # P1 mixed 域内
        tr = (sp == "train") & np.isin(src, ["suno", "fma", "jamendo"])
        va = (sp == "val") & np.isin(src, ["suno", "fma", "jamendo"])
        te_f = (sp == "test") & np.isin(src, ["suno", "fma"])
        te_j = (sp == "test") & np.isin(src, ["suno", "jamendo"])
        tests = {"vs fma": (te_f, y_gen[te_f]), "vs jamendo": (te_j, y_gen[te_j])}
        out, L, ev = run_protocol(F, sp, src, tr, va, tests, head)
        print(f"P1 mixed 域内   L*={L} val={ev*100:.2f}% | "
              + " ".join(f"{k} {v*100:.2f}%" for k, v in out.items()), flush=True)

        # P2 real 侧最坏情况:fma-only 训练 → jamendo 考
        tr2 = (sp == "train") & np.isin(src, ["suno", "fma"])
        va2 = (sp == "val") & np.isin(src, ["suno", "fma"])
        out2, L2, ev2 = run_protocol(F, sp, src, tr2, va2, tests, head)
        print(f"P2 fma-only    L*={L2} val={ev2*100:.2f}% | "
              + " ".join(f"{k} {v*100:.2f}%" for k, v in out2.items()), flush=True)

        # P3 LOGO:留一生成器(vs fma)
        logo = []
        for G in GENS:
            pool = [g for g in GENS if g != G] + ["suno"]
            tr3 = (sp == "train") & np.isin(src, pool + ["fma"])
            va3 = (sp == "val") & np.isin(src, pool + ["fma"])
            teG = np.isin(src, [G]) | ((sp == "test") & (src == "fma"))
            tests3 = {G: (teG, y_gen[teG])}
            out3, _, _ = run_protocol(F, sp, src, tr3, va3, tests3, head)
            logo.append((G, out3[G]))
        print("P3 LOGO        " + " ".join(f"{g}:{e*100:.1f}%" for g, e in logo), flush=True)
        print(f"[{head}] {time.time()-t0:.0f}s", flush=True)
    print("UPGRADE-DONE", flush=True)


if __name__ == "__main__":
    main()
