"""地图全家福矩阵(2026-08-23):每张图 × 每个陪审团全打分——"谁认谁"矩阵。

前置:latent 缓存齐 + 各源有 train 划分(resplit 内置:凡某 tag 无 train 行,
按 audio_id 哈希 70/15/15 确定性重划,写回 index.csv——与 Udio 局同法)。
训练:缺 checkpoint 的图现场训(subprocess 调 train_flow_b.py,参数全默认=与
suno/jam/udio 图同配方);已有的(suno/jam/udio)直接复用。
打分:与 score_flow_b 同一套数学,输出加 map 列;断点续跑键 (map, jury, audio_id)。

Usage(服务器 .venv-flow2,tmux):
  python flowmatch/matrix_score.py --out "$WORK/results/flow_matrix_scores.csv" --n 60
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

# 图:tag -> checkpoint(旧图复用旧路径;新图进 flow_maps/)
MAPS = {
    "suno":    "results/flow_b_suno.pt",
    "jamendo": "results/flow_b_jam.pt",
    "udio":    "results/flow_b_udio.pt",   # train-tag=udio30,udio120
    "fma":     "results/flow_maps/fma.pt",
    "acestep_v1":  "results/flow_maps/acestep_v1.pt",
    "dr1":         "results/flow_maps/dr1.pt",
    "diffrhythm2": "results/flow_maps/diffrhythm2.pt",
    "musicgen":    "results/flow_maps/musicgen.pt",
    "audioldm2":   "results/flow_maps/audioldm2.pt",
    "musicldm":    "results/flow_maps/musicldm.pt",
    "mustango":    "results/flow_maps/mustango.pt",
    "stable_audio_open": "results/flow_maps/stable_audio_open.pt",
}
TRAIN_TAG = {"udio": "udio30,udio120"}  # 其余图 train-tag = 自己名字

# 陪审团:jury -> (tags, split)
JURIES = [
    ("suno", "suno", "test"), ("jamendo", "jamendo", "test"), ("fma", "fma", "test"),
    ("ccmixter", "ccmixter", None), ("ianet", "ianet", None),
    ("udio", "udio30,udio120", "test"),
    ("acestep_v1", "acestep_v1", "test"), ("dr1", "dr1", "test"),
    ("diffrhythm2", "diffrhythm2", "test"), ("musicgen", "musicgen", "test"),
    ("audioldm2", "audioldm2", "test"), ("musicldm", "musicldm", "test"),
    ("mustango", "mustango", "test"), ("stable_audio_open", "stable_audio_open", "test"),
]


def ensure_splits(latents):
    """凡矩阵要训的 tag 没有 train 行,按 audio_id 哈希 70/15/15 重划(确定性)。"""
    idx_path = Path(latents) / "index.csv"
    rows = list(csv.DictReader(open(idx_path)))
    need = set()
    for m in MAPS:
        tags = set(TRAIN_TAG.get(m, m).split(","))
        has_train = any(r["tag"] in tags and r["split"] == "train" for r in rows)
        if not has_train:
            need |= tags
    if not need:
        print("[splits] all map tags have train rows", flush=True)
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

    # ---- 训练缺失的图 ----
    for m, ck in MAPS.items():
        ck_path = Path(work) / ck
        if ck_path.exists():
            continue
        tag = TRAIN_TAG.get(m, m)
        print(f"[train] map={m} (tag={tag}) -> {ck_path}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train_flow_b.py"),
                            "--latents", args.latents, "--train-tag", tag,
                            "--out", str(ck_path)])
        if r.returncode != 0:
            print(f"[train] map={m} FAILED, skip", flush=True)

    # ---- 矩阵打分 ----
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
                rng = random.Random(SAMPLE_SEED + hash(jury) % 10000)
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
    print("MATRIX-DONE", flush=True)


if __name__ == "__main__":
    main()
