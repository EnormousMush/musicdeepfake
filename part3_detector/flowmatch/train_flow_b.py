"""路 B:在缓存 latent 上自训小无条件流匹配速度场(2026-08-22 预注册后实现)。

模型:~6M 参数 Transformer(dim256×6层×4头),输入 (T=250, 64) latent 序列;
路径约定与 ACE-Step 完全一致:xt = t·noise + (1−t)·x0,目标 v = noise − x0,t~U(0,1)。
归一:训练集全局标量 mean/std(存进 checkpoint,打分侧同口径)。

Usage(服务器 .venv-flow2,tmux):
  python flowmatch/train_flow_b.py --latents "$WORK/latents" --train-tag suno \
    --train-split train --out "$WORK/results/flow_b_suno.pt" [--steps 12000]
镜像对照:--train-tag jamendo --train-split train --out flow_b_jam.pt
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class FlowNet(nn.Module):
    def __init__(self, dim=256, layers=6, heads=4, seq=250, feat=64):
        super().__init__()
        self.inp = nn.Linear(feat, dim)
        self.pos = nn.Parameter(torch.zeros(1, seq, dim))
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                         norm_first=True, activation="gelu", dropout=0.0)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(dim, feat)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.dim = dim

    def t_embed(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * freqs[None] * 1000.0
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x, t):
        h = self.inp(x) + self.pos[:, : x.shape[1], :]
        h = h + self.t_mlp(self.t_embed(t))[:, None, :]
        return self.out(self.blocks(h))


def load_split(latents, tag, split):
    tags = set(tag.split(","))
    idx = [r for r in csv.DictReader(open(Path(latents) / "index.csv"))
           if r["tag"] in tags and (not split or r["split"] == split)]
    X = np.stack([np.load(Path(latents) / r["path"]) for r in idx])
    return torch.from_numpy(X)  # (N,T,64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=os.path.expandvars("$WORK/latents"))
    ap.add_argument("--train-tag", default="suno")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260822)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = load_split(args.latents, args.train_tag, args.train_split)
    Xva = load_split(args.latents, args.train_tag, args.val_split)
    mu, sd = float(Xtr.mean()), float(Xtr.std())
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    print(f"train {tuple(Xtr.shape)} val {tuple(Xva.shape)} | norm mu={mu:.4f} sd={sd:.4f}", flush=True)

    net = FlowNet().to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"FlowNet {n_par/1e6:.1f}M params on {device}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    def fm_loss(x0):
        t = torch.rand(x0.shape[0], device=device)
        noise = torch.randn_like(x0)
        xt = t[:, None, None] * noise + (1 - t[:, None, None]) * x0
        v = net(xt, t)
        return ((v - (noise - x0)) ** 2).mean()

    best_val, t0 = float("inf"), time.time()
    g = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        net.train()
        ii = torch.randint(0, len(Xtr), (args.batch,), generator=g)
        loss = fm_loss(Xtr[ii].to(device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps:
            net.eval()
            with torch.no_grad():
                torch.manual_seed(7)  # 固定 val 噪声,曲线可比
                vl = float(np.mean([fm_loss(Xva[j:j+64].to(device)).item()
                                    for j in range(0, len(Xva), 64)]))
            tag = ""
            if vl < best_val:
                best_val = vl
                torch.save(dict(state=net.state_dict(), mu=mu, sd=sd,
                                step=step, val=vl, args=vars(args)), args.out)
                tag = " <== saved"
            print(f"step {step:6d} train {loss.item():.4f} val {vl:.4f} "
                  f"({time.time()-t0:.0f}s){tag}", flush=True)
    print(f"DONE best_val={best_val:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
