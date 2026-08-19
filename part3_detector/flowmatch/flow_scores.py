"""流匹配 2.0 · 零级/一级打分:ACE-Step base 速度场的过程分数。

对每个 clip(common-spec 16k/mono/10s FLAC)输出:
  S1 速度失配(Flow Mismatching 思路)= 冻住模型,对该 clip 算流匹配"训练损失":
     K 条噪声路径 × 9 个 t 档,mismatch = MSE(v̂, noise − x0);存逐 t 曲线 + 总均值;
  S2 反演三件套(InvFlowFD 思路,群体分数改造成逐 clip):
     50 步 Euler 倒走 → x̂1:先验偏离(逐帧高斯 NLL / 逐 clip 对角 FD-to-prior)
     再 50 步正走回 → 往返误差 rt_mse。
全程无条件(null_condition_emb),无 CFG,零训练。种子按 audio_id 哈希固定,可复现。

预注册(2026-08-19,vault 过程建模 §四级阶梯):
  零级读数 acestep_base/turbo vs jam+fma;一级读数 dr 族 vs 老五代;通过线见 vault。

Usage:
  ACESTEP_CHECKPOINTS_DIR="/Volumes/Seagate /honors_paper/4_models/acestep_checkpoints" \
  /Users/durunbao/Developer/ACE-Step-1.5/.venv/bin/python flow_scores.py \
    --out "/Volumes/Seagate /honors_paper/5_results/flowmatch_v2/flow_scores_L0.csv" \
    --n 50 [--juries acestep_base,acestep_turbo,jamendo,fma] [--smoke]
"""
import argparse
import csv
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import torch

ACESTEP_REPO = "/Users/durunbao/Developer/ACE-Step-1.5"
sys.path.insert(0, ACESTEP_REPO)

SEAGATE = "/Volumes/Seagate /honors_paper"
WP = f"{SEAGATE}/3_workpacks/frank-suno-round1"

# 陪审团注册表:tag -> (manifest, audio_root, source 过滤, split 过滤)
JURIES = {
    # 零级:自家孩子(本尊 v1.5-base / 蒸馏子 v1.5-turbo)
    "acestep_base":  (f"{WP}/flowmatch_export/manifest.csv", f"{WP}/flowmatch_export", "acestep_base", None),
    "acestep_turbo": (f"{WP}/flowmatch_export/manifest.csv", f"{WP}/flowmatch_export", "acestep_turbo", None),
    # 零级人类侧
    "jamendo": (f"{WP}/crossgen_add_jamendo/jamendo_manifest.csv", f"{WP}/crossgen_add_jamendo", "jamendo", "test"),
    "fma":     (f"{WP}/crossgen_export/manifest.csv", f"{WP}/crossgen_export", "fma", "test"),
    # 一级:亲戚(v1 野生 = 上一代本尊;dr 族 = 同血统)vs 跨族老五代
    "acestep_v1":  (f"{WP}/crossgen_add/manifest.csv", f"{WP}/crossgen_add", "acestep", "test"),
    "diffrhythm2": (f"{WP}/crossgen_add/manifest.csv", f"{WP}/crossgen_add", "diffrhythm2", "test"),
    "dr1":         (f"{WP}/crossgen_add_dr1/manifest.csv", f"{WP}/crossgen_add_dr1", "dr1", "test"),
    "musicgen":    (f"{WP}/crossgen_export/manifest.csv", f"{WP}/crossgen_export", "MusicGen_medium", "test"),
    "audioldm2":   (f"{WP}/crossgen_export/manifest.csv", f"{WP}/crossgen_export", "audioldm2", "test"),
    # 二级预留:suno + 拼盘
    "suno":     (f"{WP}/crossgen_export/manifest.csv", f"{WP}/crossgen_export", "suno", "test"),
    "ccmixter": (f"{WP}/heldout_export/manifest.csv", f"{WP}/heldout_export", "ccmixter", None),
    "ianet":    (f"{WP}/heldout_export/manifest.csv", f"{WP}/heldout_export", "ianet", None),
}
DEFAULT_JURIES = "acestep_base,acestep_turbo,jamendo,fma,acestep_v1,diffrhythm2,dr1,musicgen,audioldm2,suno,ccmixter,ianet"

T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_NOISE = 4      # S1 每 t 档噪声路径数
INV_STEPS = 50   # S2 反演步数(单向)
SAMPLE_SEED = 20260820  # 陪审团抽样种子


def pick_rows(tag, n):
    man, root, src, split = JURIES[tag]
    rows = [r for r in csv.DictReader(open(man))
            if r["source"] == src and (split is None or r.get("split") == split)]
    rng = random.Random(SAMPLE_SEED + hash(tag) % 10000)
    rng.shuffle(rows)
    return [(r["audio_id"], os.path.join(root, r["rel_path"])) for r in rows[:n]]


class Prober:
    def __init__(self):
        from acestep.handler import AceStepHandler
        self.h = AceStepHandler()
        status = self.h.initialize_service(
            project_root=ACESTEP_REPO, config_path="acestep-v15-base", device="auto",
            use_mlx_dit=False)  # 单份 torch DiT(16GB Mac 装不下 MLX+torch 两份)
        print("[init]", status[0].splitlines()[0], flush=True)
        if self.h.device == "mps":  # MPS 路径 dtype 写死 fp32 → 压 fp16 省一半
            self.h.model = self.h.model.half()
            self.h.dtype = torch.float16
        self.model = self.h.model
        self.device, self.dtype = self.h.device, self.h.dtype

    def encode(self, path):
        wav = self.h.process_src_audio(path)
        with torch.inference_mode():
            x0 = self.h._encode_audio_to_latents(wav)
        return x0.unsqueeze(0).to(device=self.device, dtype=self.dtype)  # (1,T,64)

    def _cond(self, x0):
        bsz, T, _ = x0.shape
        attn = torch.ones(bsz, T, device=self.device, dtype=self.dtype)
        sil = self.h.silence_latent.to(device=self.device, dtype=self.dtype)[:, :T, :].expand(bsz, -1, -1)
        context = torch.cat([sil, torch.ones_like(sil)], dim=-1)
        null_emb = self.model.null_condition_emb
        enc = null_emb.expand(bsz, 8, -1).to(device=self.device, dtype=self.dtype)
        enc_mask = torch.ones(bsz, enc.shape[1], device=self.device, dtype=self.dtype)
        return attn, enc, enc_mask, context

    def velocity(self, xt, t_scalar, cond):
        attn, enc, enc_mask, context = cond
        t = torch.full((xt.shape[0],), float(t_scalar), device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            out = self.model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t, attention_mask=attn,
                encoder_hidden_states=enc, encoder_attention_mask=enc_mask,
                context_latents=context)
        return out[0]

    def score_clip(self, audio_id, path):
        x0 = self.encode(path)
        cond = self._cond(x0)
        seed = int(hashlib.sha256(audio_id.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device="cpu").manual_seed(seed)
        row = {}
        # --- S1 速度失配 ---
        mses, coses = [], []
        for t_ in T_GRID:
            m_t = []
            for k in range(K_NOISE):
                noise = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(
                    device=self.device, dtype=self.dtype)
                xt = t_ * noise + (1.0 - t_) * x0
                v = self.velocity(xt, t_, cond).float()
                true_v = (noise - x0).float()
                m_t.append(torch.mean((v - true_v) ** 2).item())
                coses.append(torch.nn.functional.cosine_similarity(
                    v.flatten(), true_v.flatten(), dim=0).item())
            row[f"s1_mse_t{int(t_*10):02d}"] = sum(m_t) / len(m_t)
            mses.extend(m_t)
        row["s1_mse_mean"] = sum(mses) / len(mses)
        row["s1_cos_mean"] = sum(coses) / len(coses)
        # --- S2 反演 ---
        ts = torch.linspace(0.0, 1.0, INV_STEPS + 1)
        x = x0.clone()
        for i in range(INV_STEPS):
            v = self.velocity(x, ts[i].item(), cond)
            x = x + (ts[i + 1] - ts[i]).item() * v
        z = x.float()  # (1,T,64) 倒走终点,应≈N(0,I)
        row["s2_prior_nll"] = (0.5 * (z ** 2).mean()).item()  # 逐元素高斯 NLL(常数略去)
        mu = z.mean(dim=(0, 1))                                # (64,)
        var = z.var(dim=(0, 1), unbiased=False)                # (64,) 对角近似
        row["s2_fd_prior"] = ((mu ** 2).sum() + (var + 1.0 - 2.0 * var.clamp(min=0).sqrt()).sum()).item()
        for i in range(INV_STEPS, 0, -1):
            v = self.velocity(x, ts[i].item(), cond)
            x = x + (ts[i - 1] - ts[i]).item() * v
        row["s2_rt_mse"] = torch.mean((x.float() - x0.float()) ** 2).item()
        row["latent_std"] = x0.float().std().item()
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--juries", default=DEFAULT_JURIES)
    ap.add_argument("--smoke", action="store_true", help="每团 2 个 clip 干跑")
    args = ap.parse_args()
    n = 2 if args.smoke else args.n

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["jury"], r["audio_id"]) for r in csv.DictReader(open(out))}
        print(f"resume: {len(done)} scored", flush=True)

    prober = Prober()
    fields = None
    mode = "a" if out.exists() else "w"
    with open(out, mode, newline="") as f:
        w = None
        for tag in args.juries.split(","):
            rows = pick_rows(tag, n)
            print(f"[{tag}] {len(rows)} clips", flush=True)
            for audio_id, path in rows:
                if (tag, audio_id) in done:
                    continue
                t0 = time.time()
                try:
                    row = prober.score_clip(audio_id, path)
                except Exception as e:
                    print(f"  {audio_id}: ERROR {repr(e)[:80]}", flush=True)
                    continue
                row = dict(jury=tag, audio_id=audio_id, **row)
                if w is None:
                    fields = list(row.keys())
                    w = csv.DictWriter(f, fieldnames=fields)
                    if mode == "w":
                        w.writeheader()
                w.writerow(row)
                f.flush()
                print(f"  {audio_id} {time.time()-t0:.1f}s s1={row['s1_mse_mean']:.4f} "
                      f"rt={row['s2_rt_mse']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
