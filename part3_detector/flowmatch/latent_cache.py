"""路 B 前置:把陪审团音频批量编码成 ACE-Step VAE latent 缓存(服务器,GPU VAE)。

与 flow_scores_server.py 同口径:16k mono flac → 复制双声道 → 48k 重采样 → clamp
→ AutoencoderOobleck.encode.sample(种子=audio_id 哈希,钉死)→ (T,64) float32 npy。

Usage(服务器 .venv-flow2,tmux):
  python flowmatch/latent_cache.py --ckpt "$WORK/acestep_ckpt" \
    --data-root "$WORK/data_store" --out "$WORK/latents"
默认缓存:suno 全 3000(train/val/test)、jamendo 全部、fma test、拼盘全部。
"""
import argparse
import csv
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

# (tag, manifest 相对 data-root, 音频根, source, split 过滤 None=全部)
JOBS = [
    ("suno",     "crossgen_export/manifest.csv", "crossgen_export", "suno", None),
    ("jamendo",  "jamendo_manifest.csv", "crossgen_export", "jamendo", None),
    ("fma",      "crossgen_export/manifest.csv", "crossgen_export", "fma", None),
    ("ccmixter", "heldout_export/manifest.csv", "heldout_export", "ccmixter", None),
    ("ianet",    "heldout_export/manifest.csv", "heldout_export", "ianet", None),
    # Udio 决胜局(2026-08-22):SONICS devocal 语料
    ("udio30",   "devocal_export/manifest.csv", "devocal_export", "udio30_dv", None),
    ("udio120",  "devocal_export/manifest.csv", "devocal_export", "udio120_dv", None),
    ("sunov35dv", "devocal_export/manifest.csv", "devocal_export", "sunov35_dv", None),
    # 终审(2026-08-23):devocal 桥行——同一把 demucs 刀下的人类与 suno
    ("fma_dv",  "devocal_export/manifest.csv", "devocal_export", "fma", "test"),
    ("suno_dv", "devocal_export/manifest.csv", "devocal_export", "suno", "test"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expandvars("$WORK/acestep_ckpt"))
    ap.add_argument("--data-root", default=os.path.expandvars("$WORK/data_store"))
    ap.add_argument("--out", default=os.path.expandvars("$WORK/latents"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from diffusers import AutoencoderOobleck
    import torchaudio
    import soundfile as sf

    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderOobleck.from_pretrained(
        str(Path(args.ckpt) / "vae"), torch_dtype=torch.float32).to(device).eval()
    resampler = torchaudio.transforms.Resample(16000, 48000)
    print(f"[init] VAE on {device}", flush=True)

    out_root = Path(args.out)
    idx_path = out_root / "index.csv"
    done = set()
    if idx_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(idx_path))}
        print(f"resume: {len(done)}", flush=True)
    out_root.mkdir(parents=True, exist_ok=True)
    new_idx = not idx_path.exists()
    f_idx = open(idx_path, "a", newline="")
    w = csv.DictWriter(f_idx, fieldnames=["audio_id", "tag", "source", "split", "label", "path"])
    if new_idx:
        w.writeheader()

    t0, n = time.time(), 0
    for tag, man_rel, audio_rel, src, split in JOBS:
        rows = [r for r in csv.DictReader(open(Path(args.data_root) / man_rel))
                if r["source"] == src and (split is None or r.get("split") == split)]
        if args.limit:
            rows = rows[: args.limit]
        (out_root / tag).mkdir(exist_ok=True)
        print(f"[{tag}] {len(rows)} clips", flush=True)
        for r in rows:
            aid = r["audio_id"]
            if aid in done:
                continue
            path = Path(args.data_root) / audio_rel / r["rel_path"]
            try:
                y, sr = sf.read(str(path), dtype="float32", always_2d=True)
                assert sr == 16000
                audio = torch.from_numpy(y.T)
                if audio.shape[0] == 1:
                    audio = torch.cat([audio, audio], dim=0)
                audio = torch.clamp(resampler(audio[:2]), -1.0, 1.0)
                seed = int(hashlib.sha256(aid.encode()).hexdigest()[:8], 16)
                g = torch.Generator(device=device).manual_seed(seed)
                with torch.inference_mode():
                    lat = vae.encode(audio.unsqueeze(0).to(device)).latent_dist.sample(generator=g)
                lat = lat.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)  # (T,64)
                np.save(out_root / tag / f"{aid}.npy", lat)
            except Exception as e:
                print(f"  {aid}: ERROR {repr(e)[:80]}", flush=True)
                continue
            w.writerow(dict(audio_id=aid, tag=tag, source=src,
                            split=r.get("split", ""), label=r.get("label", ""),
                            path=f"{tag}/{aid}.npy"))
            f_idx.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n} cached ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE: {n} new latents", flush=True)


if __name__ == "__main__":
    main()
