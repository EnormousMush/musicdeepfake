"""流匹配 2.0 · 服务器独立版打分(2026-08-22):S1 速度失配 + S2 反演三件套。

与 Mac 版 flow_scores.py 同一套数学,但**不依赖 ACE-Step 包**——直接用
torch + transformers(trust_remote_code 加载权重目录自带的 DiT 代码)+
diffusers(AutoencoderOobleck VAE)。服务器只需在既有 venv 里补装 diffusers。

与 Mac 版的两处已知口径差(记档,不影响判读):
  1. dtype:服务器 bf16/CUDA vs Mac fp16/MPS(两边各自成表,不混行);
  2. VAE encode 的后验采样种子:服务器按 audio_id 哈希钉死(Mac 版未钉)。

Usage(服务器,tmux;铠甲先行,见 FUDAN_RUN.local.md Phase 4):
  python part3_detector/flowmatch/flow_scores_server.py --check          # 预检:陪审团/权重就位?
  python part3_detector/flowmatch/flow_scores_server.py \
    --ckpt "$WORK/acestep_ckpt" --data-root "$WORK/data_store" \
    --out "$WORK/results/flow_scores_L0_server.csv" --n 50
"""
import argparse
import csv
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

T_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
K_NOISE = 4
INV_STEPS = 50
SAMPLE_SEED = 20260820  # 与 Mac 版同种子 → 同一批陪审团样本

# tag -> (manifest 相对 data-root, 音频根相对 data-root, source 过滤, split 过滤)
# 服务器布局(2026-08-22 实勘):add-on 音频全在 crossgen_export/audio/<source>/,
# 散装 manifest(crossgen_add_manifest/add_dr1/jamendo_manifest)的 rel_path 相对 crossgen_export。
JURIES = {
    "acestep_base":  ("flowmatch_export/manifest.csv", "flowmatch_export", "acestep_base", None),
    "acestep_turbo": ("flowmatch_export/manifest.csv", "flowmatch_export", "acestep_turbo", None),
    "jamendo": ("jamendo_manifest.csv", "crossgen_export", "jamendo", "test"),
    "fma":     ("crossgen_export/manifest.csv", "crossgen_export", "fma", "test"),
    "acestep_v1":  ("crossgen_add_manifest.csv", "crossgen_export", "acestep", "test"),
    "diffrhythm2": ("crossgen_add_manifest.csv", "crossgen_export", "diffrhythm2", "test"),
    "dr1":         ("add_dr1.csv", "crossgen_export", "dr1", "test"),
    "musicgen":    ("crossgen_export/manifest.csv", "crossgen_export", "MusicGen_medium", "test"),
    "audioldm2":   ("crossgen_export/manifest.csv", "crossgen_export", "audioldm2", "test"),
    "suno":     ("crossgen_export/manifest.csv", "crossgen_export", "suno", "test"),
    "ccmixter": ("heldout_export/manifest.csv", "heldout_export", "ccmixter", None),
    "ianet":    ("heldout_export/manifest.csv", "heldout_export", "ianet", None),
}
DEFAULT_JURIES = ",".join(JURIES)


def pick_rows(data_root, tag, n):
    man_rel, audio_rel, src, split = JURIES[tag]
    man = Path(data_root) / man_rel
    if not man.exists():
        return None
    rows = [r for r in csv.DictReader(open(man))
            if r["source"] == src and (split is None or r.get("split") == split)]
    rng = random.Random(SAMPLE_SEED + hash(tag) % 10000)
    rng.shuffle(rows)
    return [(r["audio_id"], str(Path(data_root) / audio_rel / r["rel_path"])) for r in rows[:n]]


class Prober:
    def __init__(self, ckpt_dir):
        from transformers import AutoModel
        from diffusers import AutoencoderOobleck
        # 复旦 3060/驱动470 + torch2.4-cu118 的 cuDNN 初始化犯冲(CUDNN_STATUS_NOT_INITIALIZED);
        # 禁掉走原生内核,VAE 卷积略慢,DiT(注意力/线性)不受影响
        torch.backends.cudnn.enabled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        base = Path(ckpt_dir) / "acestep-v15-base"
        print(f"[init] loading DiT from {base} ({self.dtype})", flush=True)
        m = AutoModel.from_pretrained(
            str(base), trust_remote_code=True, torch_dtype=self.dtype)  # 先落 CPU
        # 瘦身:探针只用 decoder + null_condition_emb;encoder/tokenizer/detokenizer
        # 共 818M 参数(bf16 1.6GB)全扔——GPU1 与同学的任务合租,余量只有 ~5.8GB
        for dead in ("encoder", "tokenizer", "detokenizer"):
            if hasattr(m, dead):
                setattr(m, dead, None)
        self.model = m.to(self.device).eval()
        print("[init] loading VAE (AutoencoderOobleck, CPU fp32)", flush=True)
        # VAE 留 CPU:GPU 与同学合租余量小,且 cuDNN 禁用后原生卷积单次要 822MB;
        # 10s clip 的 CPU encode 仅数秒,换 GPU 只伺候 decoder
        self.vae = AutoencoderOobleck.from_pretrained(
            str(Path(ckpt_dir) / "vae"), torch_dtype=torch.float32).eval()
        sl = torch.load(str(base / "silence_latent.pt"), map_location="cpu")
        if sl.dim() == 3 and sl.shape[1] == 64 and sl.shape[2] != 64:
            sl = sl.transpose(1, 2)  # 盘上存的是 (1,64,T) 通道在前;handler 加载时会转置,独立版自己转
        self.silence_latent = sl.to(device=self.device, dtype=self.dtype)  # (1,T0,64)
        import torchaudio
        self.resampler = torchaudio.transforms.Resample(16000, 48000)

    def encode(self, path, seed):
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=True)  # (N,C)
        assert sr == 16000, f"expect 16k common-spec, got {sr}"
        audio = torch.from_numpy(y.T)                # (C,N)
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)  # mono → 双声道(同 handler)
        audio = self.resampler(audio[:2])
        audio = torch.clamp(audio, -1.0, 1.0)
        vae_in = audio.unsqueeze(0).float()                       # CPU fp32
        g = torch.Generator().manual_seed(seed)
        with torch.inference_mode():
            latents = self.vae.encode(vae_in).latent_dist.sample(generator=g)  # (1,64,T)
        return latents.transpose(1, 2).to(self.device).to(self.dtype)  # (1,T,64) → GPU bf16

    def _cond(self, x0):
        bsz, T, _ = x0.shape
        attn = torch.ones(bsz, T, device=self.device, dtype=self.dtype)
        sil = self.silence_latent[:, :T, :].expand(bsz, -1, -1)
        context = torch.cat([sil, torch.ones_like(sil)], dim=-1)
        enc = self.model.null_condition_emb.expand(bsz, 8, -1).to(
            device=self.device, dtype=self.dtype)
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
        seed = int(hashlib.sha256(audio_id.encode()).hexdigest()[:8], 16)
        x0 = self.encode(path, seed)
        cond = self._cond(x0)
        g = torch.Generator(device="cpu").manual_seed(seed)
        row = {}
        mses, coses = [], []
        for t_ in T_GRID:
            m_t = []
            for _ in range(K_NOISE):
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
        ts = torch.linspace(0.0, 1.0, INV_STEPS + 1)
        x = x0.clone()
        for i in range(INV_STEPS):
            v = self.velocity(x, ts[i].item(), cond)
            x = x + (ts[i + 1] - ts[i]).item() * v
        z = x.float()
        row["s2_prior_nll"] = (0.5 * (z ** 2).mean()).item()
        mu = z.mean(dim=(0, 1))
        var = z.var(dim=(0, 1), unbiased=False)
        row["s2_fd_prior"] = ((mu ** 2).sum() + (var + 1.0 - 2.0 * var.clamp(min=0).sqrt()).sum()).item()
        for i in range(INV_STEPS, 0, -1):
            v = self.velocity(x, ts[i].item(), cond)
            x = x + (ts[i - 1] - ts[i]).item() * v
        row["s2_rt_mse"] = torch.mean((x.float() - x0.float()) ** 2).item()
        row["latent_std"] = x0.float().std().item()
        return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expandvars("$WORK/acestep_ckpt"))
    ap.add_argument("--data-root", default=os.path.expandvars("$WORK/data_store"))
    ap.add_argument("--out", default=os.path.expandvars("$WORK/results/flow_scores_L0_server.csv"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--juries", default=DEFAULT_JURIES)
    ap.add_argument("--check", action="store_true", help="预检:不加载模型,只查数据/权重就位")
    args = ap.parse_args()

    if args.check:
        ok = True
        for f in ["acestep-v15-base/model.safetensors", "acestep-v15-base/silence_latent.pt",
                  "vae/diffusion_pytorch_model.safetensors"]:
            p = Path(args.ckpt) / f
            print(("  ✅" if p.exists() else "  ❌") + f" ckpt/{f}")
            ok &= p.exists()
        for tag in args.juries.split(","):
            rows = pick_rows(args.data_root, tag, 3)
            if rows is None:
                print(f"  ❌ jury {tag}: manifest 缺失"); ok = False; continue
            missing = [aid for aid, p in rows if not os.path.exists(p)]
            print(f"  {'✅' if rows and not missing else '❌'} jury {tag}: {len(rows)} sampled"
                  + (f", missing audio e.g. {missing[:1]}" if missing else ""))
            ok &= bool(rows) and not missing
        try:
            import diffusers  # noqa
            print(f"  ✅ diffusers {diffusers.__version__}")
        except ImportError:
            print("  ❌ diffusers 未安装:pip install 'diffusers>=0.30'"); ok = False
        print("CHECK", "PASS" if ok else "FAIL")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["jury"], r["audio_id"]) for r in csv.DictReader(open(out))}
        print(f"resume: {len(done)} scored", flush=True)
    prober = Prober(args.ckpt)
    mode = "a" if out.exists() else "w"
    with open(out, mode, newline="") as f:
        w = None
        for tag in args.juries.split(","):
            rows = pick_rows(args.data_root, tag, args.n)
            if rows is None:
                print(f"[{tag}] SKIP(manifest 缺失)", flush=True)
                continue
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
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if mode == "w":
                        w.writeheader()
                w.writerow(row)
                f.flush()
                print(f"  {audio_id} {time.time()-t0:.1f}s s1={row['s1_mse_mean']:.4f} "
                      f"rt={row['s2_rt_mse']:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
