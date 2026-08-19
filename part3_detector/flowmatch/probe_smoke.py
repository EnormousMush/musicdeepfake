"""流匹配 2.0 · Phase 0 冒烟测试:验三件命脉。

① AceStepHandler 初始化(acestep-v15-base,Mac 上 device 自动)
② VAE encode:common-spec FLAC → 25Hz/64 维 latent
③ 裸速度场:decoder(xt, t, 无条件) → v̂,对比几何真速度 (noise − x0)
④ 迷你反演:10 步 Euler 倒走→正走往返,报告重建误差

用法(用 ACE-Step 的 venv 跑):
  ACESTEP_CHECKPOINTS_DIR="/Volumes/Seagate /honors_paper/4_models/acestep_checkpoints" \
  /Users/durunbao/Developer/ACE-Step-1.5/.venv/bin/python probe_smoke.py <一个16k mono flac>
"""
import os
import sys
import traceback

import torch

ACESTEP_REPO = "/Users/durunbao/Developer/ACE-Step-1.5"
sys.path.insert(0, ACESTEP_REPO)

CKPT_DIR = os.environ.get(
    "ACESTEP_CHECKPOINTS_DIR",
    "/Volumes/Seagate /honors_paper/4_models/acestep_checkpoints",
)


def main():
    flac = sys.argv[1]
    from acestep.handler import AceStepHandler

    h = AceStepHandler()
    # ① 初始化(不带 LM;config 用 base 版 DiT)
    status = h.initialize_service(
        project_root=ACESTEP_REPO,
        config_path="acestep-v15-base",
        device="auto",
        use_mlx_dit=False,  # 只留 torch 一份 DiT(16GB 机器装不下两份)
    )
    print("[init]", status[0].splitlines()[0])
    # MPS 路径 dtype 写死 fp32(2B 参数=8GB);压到 fp16 省一半
    if h.device == "mps":
        h.model = h.model.half()
        h.dtype = torch.float16
    model = h.model
    device, dtype = h.device, h.dtype
    print("[init] device:", device, "dtype:", dtype)
    print("[init] has decoder:", hasattr(model, "decoder"),
          "| null_cond:", hasattr(model, "null_condition_emb"),
          "| silence_latent:", getattr(h, "silence_latent", None) is not None)

    # ② VAE encode
    wav = h.process_src_audio(flac)  # -> (channels, samples) 预处理到模型采样率
    print("[vae] processed audio shape:", tuple(wav.shape))
    with torch.inference_mode():
        x0 = h._encode_audio_to_latents(wav)  # (T, 64)
    print("[vae] latent shape:", tuple(x0.shape), "mean=%.3f std=%.3f" %
          (x0.float().mean(), x0.float().std()))

    x0 = x0.unsqueeze(0).to(device=device, dtype=dtype)  # (1, T, 64)
    bsz, T, C = x0.shape
    attn = torch.ones(bsz, T, device=device, dtype=dtype)

    # 无条件条件组装:null_condition_emb + silence context
    silence = h.silence_latent.to(device=device, dtype=dtype)  # (1, L, 64)
    sil = silence[:, :T, :].expand(bsz, -1, -1)
    chunk = torch.ones_like(sil)  # 全 1 = 全段生成(text2music 语义)
    context = torch.cat([sil, chunk], dim=-1)  # (1, T, 128)
    null_emb = model.null_condition_emb  # (1, 1, D) 或类似
    print("[cond] null_condition_emb shape:", tuple(null_emb.shape))
    enc_states = null_emb.expand(bsz, 8, -1).to(device=device, dtype=dtype)
    enc_mask = torch.ones(bsz, enc_states.shape[1], device=device, dtype=dtype)

    def velocity(xt, t_scalar):
        t = torch.full((bsz,), t_scalar, device=device, dtype=dtype)
        with torch.inference_mode():
            out = model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attn,
                encoder_hidden_states=enc_states,
                encoder_attention_mask=enc_mask,
                context_latents=context,
            )
        return out[0]

    # ③ 单点速度失配(S1 原型)
    torch.manual_seed(0)
    noise = torch.randn_like(x0)
    for t_ in (0.25, 0.5, 0.75):
        xt = t_ * noise + (1.0 - t_) * x0
        v = velocity(xt, t_)
        true_v = noise - x0
        mse = torch.mean((v.float() - true_v.float()) ** 2).item()
        cos = torch.nn.functional.cosine_similarity(
            v.float().flatten(), true_v.float().flatten(), dim=0).item()
        print(f"[S1] t={t_:.2f} v̂ shape={tuple(v.shape)} mismatch_mse={mse:.4f} cos={cos:.3f}")

    # ④ 迷你往返反演(10 步 Euler:x0 →(加速度场倒走)→ x1 → 正走 → x0')
    steps = 10
    ts = torch.linspace(0.0, 1.0, steps + 1)
    x = x0.clone()
    for i in range(steps):  # 倒走:t 从 0 → 1
        t_c, t_n = ts[i].item(), ts[i + 1].item()
        v = velocity(x, t_c)
        x = x + (t_n - t_c) * v
    x1_hat = x
    print("[S2] inverted 'noise' std=%.3f (先验应≈1)" % x1_hat.float().std().item())
    for i in range(steps, 0, -1):  # 正走:t 从 1 → 0
        t_c, t_n = ts[i].item(), ts[i - 1].item()
        v = velocity(x, t_c)
        x = x + (t_n - t_c) * v
    rt_err = torch.mean((x.float() - x0.float()) ** 2).item()
    print(f"[S2] round-trip mse={rt_err:.5f}")
    print("SMOKE OK")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
