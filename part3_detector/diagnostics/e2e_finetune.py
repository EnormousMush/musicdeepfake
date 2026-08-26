"""议题四:端到端微调实验(2026-08-26 预注册;补 E11 的 scope 缺口)。

E11 只覆盖"冻结特征上的一切头";本实验把 MERT-95M 整个解冻微调,跑 E11 考卷
里最关键的两张:
  P2   fma-only real 侧最坏情况(suno+fma train → test vs fma / vs jamendo)
  P3   LOGO 留一生成器(其余 4 家 pool + suno + fma train → 留出家 vs fma-test)
预注册赌注(沿 E11 逻辑):域内更饱和,但 real 侧退化与 LOGO 塌陷**依旧存活
甚至更糟**——若成立,"容量不解决域偏移且加重"证据链在端到端处闭环。

配置:mean-pool 最后一层 + 线性头;AdamW(骨干 1e-5 / 头 1e-3),AMP fp16,
batch 4 × grad-accum 4,3 epochs,按 val EER 选 epoch;共享 3060(~5.8GB)可容。
断点续跑:每个 run 出 results/e2e_<run>.json,存在即跳过。
组件注记:E12 占位符原则适用——本实验测的是"端到端微调"这条路线,不是终版组件。

Usage(服务器老 .venv/torch1.12,HF 离线护甲,tmux):
  python diagnostics/e2e_finetune.py --data-dir data_store/crossgen_export --inst-only
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

GENS = ["MusicGen_medium", "audioldm2", "musicldm", "mustango", "stable_audio_open"]
SEED = 20260826
SR = 24000


def load_rows(args):
    data_dir = Path(args.data_dir)
    keep = set(["suno", "fma", "jamendo"] + GENS)
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv")) if r["source"] in keep]
    if args.inst_only:
        ids = instrumental_ids()
        if ids is not None:
            rows = filter_rows(rows, ids)
    for r in rows:
        r["path"] = str(data_dir / r["rel_path"])
    return rows


def load_wav(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = (y - y.mean()) / (y.std() + 1e-7)   # MERT 前端的逐条归一化(Wav2Vec2FeatureExtractor 同款)
    return y.astype(np.float32)


class E2E(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.backbone(input_values=x).last_hidden_state  # [B,T,D]
        return self.head(h.mean(dim=1)).squeeze(-1)


def run_one(name, tr_rows, va_rows, tests, args, device):
    out_path = Path(os.path.expandvars("$WORK")) / "results" / f"e2e_{name}.json"
    if out_path.exists():
        print(f"[{name}] exists, skip", flush=True)
        return
    from transformers import AutoModel
    torch.manual_seed(SEED)
    backbone = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()   # 共享 3060 只有 ~5.8GB,拿算力换显存
    net = E2E(backbone, backbone.config.hidden_size).to(device)
    opt = torch.optim.AdamW([
        {"params": net.backbone.parameters(), "lr": 1e-5},
        {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    npos = sum(1 for r in tr_rows if r["source"] != "fma")
    nneg = len(tr_rows) - npos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([max(1.0, nneg / max(1, npos))], device=device))
    print(f"[{name}] train {len(tr_rows)} (pos {npos}) val {len(va_rows)}", flush=True)

    rng = np.random.default_rng(SEED)

    def batches(rows, bs, shuffle):
        idx = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
        for i in range(0, len(idx), bs):
            chunk = [rows[j] for j in idx[i:i + bs]]
            wavs = [load_wav(r["path"]) for r in chunk]
            T = min(len(w) for w in wavs)
            x = torch.from_numpy(np.stack([w[:T] for w in wavs])).to(device)
            y = torch.tensor([float(r["source"] != "fma") for r in chunk], device=device)
            yield x, y

    @torch.no_grad()
    def score(rows):
        net.eval()
        out = []
        for x, _ in batches(rows, args.batch * 2, False):
            with torch.cuda.amp.autocast():
                out.append(torch.sigmoid(net(x)).float().cpu().numpy())
        return np.concatenate(out)

    best = (None, 1.0)
    for ep in range(args.epochs):
        net.train()
        t0, step = time.time(), 0
        opt.zero_grad()
        for x, y in batches(tr_rows, args.batch, True):
            with torch.cuda.amp.autocast():
                loss = lossf(net(x), y) / args.accum
            scaler.scale(loss).backward()
            step += 1
            if step % args.accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sva = score(va_rows)
        yva = np.array([float(r["source"] != "fma") for r in va_rows])
        ev = compute_eer(yva, sva)["eer"]
        print(f"[{name}] ep{ep} val EER {ev*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
        if ev < best[1]:
            best = ({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ev)
    net.load_state_dict(best[0])
    res = {"run": name, "val_eer": best[1]}
    for tname, rows in tests.items():
        s = score(rows)
        y = np.array([float(r["source"] != "fma" and r["source"] != "jamendo") for r in rows])
        res[tname] = compute_eer(y, s)["eer"]
        print(f"[{name}] {tname}: {res[tname]*100:.2f}%", flush=True)
    out_path.write_text(json.dumps(res, indent=1))
    del net, backbone
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--inst-only", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="dry-run cap per split")
    args = ap.parse_args()
    device = "cuda"
    rows = load_rows(args)

    def pick(srcs, split):
        rr = [r for r in rows if r["source"] in srcs and r["split"] == split]
        return rr[: args.limit] if args.limit else rr

    # 生成器侧 pool/eval 划分(与 E11 P3 同法:80/20 by hash 已在 drill 用过;
    # 此处沿 E11 原口径:全部 test 行做池,留出家用全部行考)
    import hashlib
    def gen_pool(g):
        rr = [r for r in rows if r["source"] == g]
        return [r for r in rr if int(hashlib.md5(r["audio_id"].encode()).hexdigest()[:8], 16) % 100 < 80]

    # P2: fma-only worst case
    tr = pick(["suno", "fma"], "train")
    va = pick(["suno", "fma"], "val")
    tests = {"vs_fma": pick(["suno", "fma"], "test"),
             "vs_jam": pick(["suno"], "test") + [r for r in rows if r["source"] == "jamendo" and r["split"] == "test"]}
    run_one("P2_fmaonly", tr, va, tests, args, device)

    # P3 LOGO
    for G in GENS:
        pool = []
        for g in GENS:
            if g != G:
                pool += gen_pool(g)
        tr = pool + pick(["suno", "fma"], "train")
        va = pick(["suno", "fma"], "val")
        heldout = [r for r in rows if r["source"] == G] + pick(["fma"], "test")
        run_one(f"LOGO_{G}", tr, va, {"heldout": heldout}, args, device)
    print("E2E-DONE", flush=True)


if __name__ == "__main__":
    main()
