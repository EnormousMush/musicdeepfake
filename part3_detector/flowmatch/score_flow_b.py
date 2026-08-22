"""路 B 打分:用自训小流模型对缓存 latent 出过程分数(与零级同一套数学,无条件无 CFG)。

主力分数 = 先验接近度(prior_nll / fd_prior,零级教训);s1 失配、往返照算备查。
陪审团直接读 latent 缓存(index.csv),无需再碰音频。

Usage(服务器 .venv-flow2):
  python flowmatch/score_flow_b.py --model "$WORK/results/flow_b_suno.pt" \
    --latents "$WORK/latents" --out "$WORK/results/flow_b_scores.csv" --n 50
读数:suno(test) vs jamendo(test)+fma(test);拼盘全量;附 acestep 参考。
"""
import argparse
import csv
import hashlib
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from train_flow_b import FlowNet

T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_NOISE = 4
INV_STEPS = 50
SAMPLE_SEED = 20260820

# (jury, tag, split 过滤;n=None 全量)
JURIES = [
    ("suno_test", "suno", "test"),
    ("jamendo_test", "jamendo", "test"),
    ("fma_test", "fma", "test"),
    ("ccmixter", "ccmixter", None),
    ("ianet", "ianet", None),
    ("udio_test", "udio30,udio120", "test"),     # Udio 决胜局主角
    ("sunov35dv_test", "sunov35dv", "test"),     # devocal 混淆对照
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--latents", default=os.path.expandvars("$WORK/latents"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.model, map_location=device)
    net = FlowNet().to(device).eval()
    net.load_state_dict(ck["state"])
    mu, sd = ck["mu"], ck["sd"]
    print(f"[init] model step={ck['step']} val={ck['val']:.4f} norm=({mu:.4f},{sd:.4f})", flush=True)

    rows_idx = list(csv.DictReader(open(Path(args.latents) / "index.csv")))
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["jury"], r["audio_id"]) for r in csv.DictReader(open(out))}
        print(f"resume: {len(done)}", flush=True)
    mode = "a" if out.exists() else "w"

    @torch.no_grad()
    def velocity(x, t_scalar):
        t = torch.full((x.shape[0],), float(t_scalar), device=device)
        return net(x, t)

    with open(out, mode, newline="") as f:
        w = None
        for jury, tag, split in JURIES:
            tags = set(tag.split(","))
            rows = [r for r in rows_idx if r["tag"] in tags and (not split or r["split"] == split)]
            rng = random.Random(SAMPLE_SEED + hash(jury) % 10000)
            rng.shuffle(rows)
            rows = rows[: args.n]
            print(f"[{jury}] {len(rows)}", flush=True)
            for r in rows:
                aid = r["audio_id"]
                if (jury, aid) in done:
                    continue
                t0 = time.time()
                x0 = torch.from_numpy(np.load(Path(args.latents) / r["path"])).unsqueeze(0)
                x0 = ((x0 - mu) / sd).to(device)
                seed = int(hashlib.sha256(aid.encode()).hexdigest()[:8], 16)
                g = torch.Generator().manual_seed(seed)
                row = {}
                mses, coses = [], []
                for t_ in T_GRID:
                    m_t = []
                    for _ in range(K_NOISE):
                        noise = torch.randn(x0.shape, generator=g).to(device)
                        xt = t_ * noise + (1.0 - t_) * x0
                        v = velocity(xt, t_)
                        tv = noise - x0
                        m_t.append(((v - tv) ** 2).mean().item())
                        coses.append(torch.nn.functional.cosine_similarity(
                            v.flatten(), tv.flatten(), dim=0).item())
                    row[f"s1_mse_t{int(t_*10):02d}"] = sum(m_t) / len(m_t)
                    mses.extend(m_t)
                row["s1_mse_mean"] = sum(mses) / len(mses)
                row["s1_cos_mean"] = sum(coses) / len(coses)
                ts = torch.linspace(0.0, 1.0, INV_STEPS + 1)
                x = x0.clone()
                for i in range(INV_STEPS):
                    x = x + (ts[i + 1] - ts[i]).item() * velocity(x, ts[i].item())
                z = x
                row["s2_prior_nll"] = (0.5 * (z ** 2).mean()).item()
                m_ = z.mean(dim=(0, 1)); va = z.var(dim=(0, 1), unbiased=False)
                row["s2_fd_prior"] = ((m_ ** 2).sum() + (va + 1.0 - 2.0 * va.clamp(min=0).sqrt()).sum()).item()
                for i in range(INV_STEPS, 0, -1):
                    x = x + (ts[i - 1] - ts[i]).item() * velocity(x, ts[i].item())
                row["s2_rt_mse"] = ((x - x0) ** 2).mean().item()
                row = dict(jury=jury, audio_id=aid, **row)
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if mode == "w":
                        w.writeheader()
                w.writerow(row); f.flush()
                print(f"  {aid} {time.time()-t0:.1f}s nll={row['s2_prior_nll']:.4f} "
                      f"fd={row['s2_fd_prior']:.2f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
