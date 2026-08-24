"""弊端一判决实验:版本半衰期曲线(2026-08-24 预注册;实验2论文方向 议题一)。

问题:权重级签名下,生成器版本升级后旧档案掉多少检出力?
证据A(自认力梯度)vs 证据B(家族泛音)谁赢——
预注册判读:跨版本 EER ≤30% 且显著低于人类图 50% 基线 → 泛音赢(半盲);
≥40% → 梯度赢(接近全瞎),保鲜制度升格框架核心组件。

设计:Suno 版本阶梯 sunov2dv / sunov3dv / sunov35dv(SONICS devocal,latent 已缓存)
各训一张图(配方与矩阵全同),加主库 suno 图,4 图 × 8 陪审团全打分。
devocal 出处排雷照搬 Udio 两段式协议:devocal 陪审团的 real 侧参照用 fma_dv
(同一把 demucs 刀),原生陪审团用 fma/jamendo。

与 matrix_score 的差别:陪审团抽样种子改用 md5(修 hash() 进程盐 bug),
故本局所有 (map, jury) 均在本进程内重打分,内部对齐。

Usage(服务器 .venv-flow2,tmux):
  python flowmatch/version_halflife.py --out "$WORK/results/flow_halflife_scores.csv" --n 60
"""
import argparse
import csv
import hashlib
import os
import random
import subprocess
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
SAMPLE_SEED = 20260820

MAPS = {
    "suno":      "results/flow_b_suno.pt",          # 主库(较新版本,原生 instrumental)
    "sunov35dv": "results/flow_maps/sunov35dv.pt",
    "sunov3dv":  "results/flow_maps/sunov3dv.pt",
    "sunov2dv":  "results/flow_maps/sunov2dv.pt",
}

# jury -> (tags, split)
JURIES = [
    ("suno", "suno", "test"),
    ("sunov35dv", "sunov35dv", "test"), ("sunov3dv", "sunov3dv", "test"),
    ("sunov2dv", "sunov2dv", "test"),
    ("suno_dv", "suno_dv", None), ("fma_dv", "fma_dv", None),   # devocal 桥
    ("fma", "fma", "test"), ("jamendo", "jamendo", "test"),      # 原生 real 侧
]


def ensure_splits(latents):
    """版本 tag 若无 train 行,按 audio_id md5 70/15/15 确定性重划(与矩阵同法)。"""
    idx_path = Path(latents) / "index.csv"
    rows = list(csv.DictReader(open(idx_path)))
    need = set()
    for m in MAPS:
        if m == "suno":
            continue
        if not any(r["tag"] == m and r["split"] == "train" for r in rows):
            need.add(m)
    if not need:
        print("[splits] all version tags have train rows", flush=True)
        return
    n = 0
    for r in rows:
        if r["tag"] in need:
            h = int(hashlib.md5(r["audio_id"].encode()).hexdigest()[:8], 16) % 100
            r["split"] = "train" if h < 70 else ("val" if h < 85 else "test")
            n += 1
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"[splits] resplit {sorted(need)}: {n} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default=os.path.expandvars("$WORK/latents"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    work = os.path.expandvars("$WORK")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_splits(args.latents)
    Path(work, "results/flow_maps").mkdir(parents=True, exist_ok=True)

    rows_idx = list(csv.DictReader(open(Path(args.latents) / "index.csv")))
    for m in MAPS:
        if m == "suno":
            continue
        cnt = {}
        for r in rows_idx:
            if r["tag"] == m:
                cnt[r["split"]] = cnt.get(r["split"], 0) + 1
        print(f"[data] {m}: {cnt}", flush=True)

    # ---- 训练缺失的版本图(配方=矩阵默认) ----
    for m, ck in MAPS.items():
        ck_path = Path(work) / ck
        if ck_path.exists():
            continue
        print(f"[train] map={m} -> {ck_path}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train_flow_b.py"),
                            "--latents", args.latents, "--train-tag", m,
                            "--out", str(ck_path)])
        if r.returncode != 0:
            print(f"[train] map={m} FAILED, skip", flush=True)

    # ---- 打分(数学与矩阵逐字同) ----
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
                jseed = SAMPLE_SEED + int(hashlib.md5(jury.encode()).hexdigest()[:8], 16) % 10000
                rng = random.Random(jseed)
                rng.shuffle(rows)
                rows = rows[: args.n]
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
    print("HALFLIFE-DONE", flush=True)


if __name__ == "__main__":
    main()
