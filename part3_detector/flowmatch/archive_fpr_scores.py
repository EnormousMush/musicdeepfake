"""弊端三 rigorous 版前置:10 张 AI 图 × 全量陪审团打分(2026-08-25)。

矩阵局(E10)每团只抽 60 首,校准集只剩 114 首人类歌,撑不住 Šidák 极端分位。
本局铺满:人类四语料全部划分(fma/jam 各 ~3000 + ccmixter/ianet 全部)+
各 AI 陪审团全测试集。数学与 matrix_score 逐字同;断点续跑键 (map,jury,audio_id)
——先把 flow_matrix_scores.csv 拷成本局输出,已打过的 60 首/团直接跳过。

Usage(服务器 .venv-flow2,tmux):
  cp "$WORK/results/flow_matrix_scores.csv" "$WORK/results/flow_fpr_scores.csv"
  python flowmatch/archive_fpr_scores.py --out "$WORK/results/flow_fpr_scores.csv"
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_flow_b import FlowNet

T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_NOISE = 4
INV_STEPS = 50

# 10 张 AI 图 + 2 张人类图(2026-08-26 追加:双图 LR 需要人类参照图的全量分数;
# 人类图不参与误伤模拟,只当 LR 分母)
MAPS = {
    "jamendo": "results/flow_b_jam.pt",
    "fma":     "results/flow_maps/fma.pt",
    "suno":    "results/flow_b_suno.pt",
    "udio":    "results/flow_b_udio.pt",
    "acestep_v1":  "results/flow_maps/acestep_v1.pt",
    "dr1":         "results/flow_maps/dr1.pt",
    "diffrhythm2": "results/flow_maps/diffrhythm2.pt",
    "musicgen":    "results/flow_maps/musicgen.pt",
    "audioldm2":   "results/flow_maps/audioldm2.pt",
    "musicldm":    "results/flow_maps/musicldm.pt",
    "mustango":    "results/flow_maps/mustango.pt",
    "stable_audio_open": "results/flow_maps/stable_audio_open.pt",
}

# jury -> (tags, split);人类全划分(AI 图没见过任何人类歌,全部合法),AI 全测试集
JURIES = [
    ("fma", "fma", None), ("jamendo", "jamendo", None),
    ("ccmixter", "ccmixter", None), ("ianet", "ianet", None),
    ("suno", "suno", "test"), ("udio", "udio30,udio120", "test"),
    ("acestep_v1", "acestep_v1", "test"), ("dr1", "dr1", "test"),
    ("diffrhythm2", "diffrhythm2", "test"), ("musicgen", "musicgen", "test"),
    ("audioldm2", "audioldm2", "test"), ("musicldm", "musicldm", "test"),
    ("mustango", "mustango", "test"), ("stable_audio_open", "stable_audio_open", "test"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=os.path.expandvars("$WORK/latents"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="per-(map,jury) cap for dry run")
    args = ap.parse_args()
    work = os.path.expandvars("$WORK")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows_idx = list(csv.DictReader(open(Path(args.latents) / "index.csv")))
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["map"], r["jury"], r["audio_id"]) for r in csv.DictReader(open(out))}
        print(f"resume: {len(done)}", flush=True)
    mode = "a" if out.exists() else "w"

    with open(out, mode, newline="") as f:
        w = None
        for m, ck in MAPS.items():
            ck_path = Path(work) / ck
            if not ck_path.exists():
                print(f"[score] map={m} no checkpoint, skip", flush=True)
                continue
            ckpt = torch.load(str(ck_path), map_location=device)
            net = FlowNet().to(device).eval()
            net.load_state_dict(ckpt["state"])
            mu, sd = ckpt["mu"], ckpt["sd"]
            print(f"[score] map={m} (step={ckpt['step']} val={ckpt['val']:.4f})", flush=True)

            @torch.no_grad()
            def velocity(x, t_scalar):
                t = torch.full((x.shape[0],), float(t_scalar), device=device)
                return net(x, t)

            for jury, tags_s, split in JURIES:
                tags = set(tags_s.split(","))
                rows = [r for r in rows_idx if r["tag"] in tags and (not split or r["split"] == split)]
                if args.limit:
                    rows = rows[: args.limit]
                todo = [r for r in rows if (m, jury, r["audio_id"]) not in done]
                if not todo:
                    continue
                t0 = time.time()
                for r in todo:
                    aid = r["audio_id"]
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
                    row = dict(map=m, jury=jury, audio_id=aid, **row)
                    if w is None:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if mode == "w":
                            w.writeheader()
                    w.writerow(row)
                f.flush()
                print(f"  {m} x {jury}: {len(todo)} clips ({time.time()-t0:.0f}s)", flush=True)
    print("FPR-SCORES-DONE", flush=True)


if __name__ == "__main__":
    main()
