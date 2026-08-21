"""流匹配 2.0 · 零级自产语料:ACE-Step 自生成 50 首 base + 50 首 turbo。

设计(预注册 2026-08-19):
- 8 genre × 6-7 首,纯器乐(caption 带 instrumental,无歌词),与 suno 语料 genre 谱对齐;
- base 版生成 = 探针本尊的孩子(最纯阳性对照);turbo 版 = 蒸馏子(零级内的微型家族阶梯);
- thinking=False、llm_handler=None:纯 DiT 流匹配产物,不掺 LM 语义码;
- 时长 30s、固定种子表(可复现),生成后由 selfcorpus_export.py 切正中 10s 转 common-spec。

用法:
  ACESTEP_CHECKPOINTS_DIR="/Volumes/Seagate /honors_paper/4_models/acestep_checkpoints" \
  /Users/durunbao/Developer/ACE-Step-1.5/.venv/bin/python gen_selfcorpus.py --variant base
  （--variant turbo 跑第二轮;--limit N 冒烟用）
"""
import argparse
import csv
import os
import sys
import time

ACESTEP_REPO = "/Users/durunbao/Developer/ACE-Step-1.5"
sys.path.insert(0, ACESTEP_REPO)

OUT_ROOT = "/Volumes/Seagate /honors_paper/2_corpora_ai/acestep_self"

GENRES = ["pop", "rock", "jazz", "blues", "electronic", "classical", "hiphop", "country"]
# 每 genre 6-7 首凑 50;caption 模板刻意朴素多样,不堆修饰词
CAPTIONS = {
    "pop":        ["upbeat instrumental pop, catchy synth melody",
                   "mellow instrumental pop ballad, piano and strings",
                   "bright instrumental pop, acoustic guitar and claps",
                   "dreamy instrumental synth pop",
                   "energetic instrumental dance pop",
                   "warm instrumental pop, electric piano groove",
                   "instrumental pop anthem, big drums"],
    "rock":       ["instrumental rock, distorted guitars and driving drums",
                   "slow instrumental blues rock, expressive guitar solo",
                   "instrumental indie rock, jangly guitars",
                   "heavy instrumental hard rock riff",
                   "instrumental surf rock, twangy guitar",
                   "instrumental post-rock build-up, ambient guitars"],
    "jazz":       ["instrumental jazz trio, piano bass and brushes",
                   "smooth instrumental jazz, saxophone lead",
                   "uptempo instrumental bebop, trumpet and piano",
                   "instrumental latin jazz, congas and horns",
                   "late night instrumental jazz ballad",
                   "instrumental gypsy jazz, fast acoustic guitar"],
    "blues":      ["slow instrumental blues, electric guitar and organ",
                   "instrumental delta blues, slide guitar",
                   "uptempo instrumental jump blues, horns",
                   "instrumental chicago blues shuffle, harmonica",
                   "instrumental blues jam, guitar and piano trading",
                   "smoky instrumental blues, muted trumpet"],
    "electronic": ["instrumental deep house, warm bassline",
                   "instrumental techno, hypnotic percussion",
                   "instrumental ambient electronica, evolving pads",
                   "instrumental drum and bass, fast breaks",
                   "instrumental synthwave, retro arpeggios",
                   "instrumental melodic dubstep, heavy drop"],
    "classical":  ["solo piano piece, romantic style",
                   "string quartet, elegant and flowing",
                   "orchestral piece, dramatic and cinematic",
                   "baroque style harpsichord piece",
                   "gentle classical guitar etude",
                   "minimalist piano and strings"],
    "hiphop":     ["instrumental boom bap hip hop beat, jazzy samples",
                   "instrumental trap beat, 808 bass",
                   "instrumental lofi hip hop, dusty piano",
                   "instrumental west coast hip hop groove",
                   "instrumental drill beat, dark strings",
                   "instrumental old school hip hop, funky break"],
    "country":    ["instrumental country, acoustic guitar and fiddle",
                   "instrumental bluegrass, fast banjo",
                   "instrumental country ballad, pedal steel guitar",
                   "instrumental outlaw country groove",
                   "instrumental country rock, twangy telecaster",
                   "instrumental western swing, fiddle and guitar"],
}
DURATION_S = 15  # 只取正中 10s;M1 上 30s×30 步撞 600s 超时,15s 省一半
BASE_SEED = 20260819
os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT", "1800")  # M1 慢,放宽内部超时


def build_joblist():
    jobs, i = [], 0
    for g in GENRES:
        for cap in CAPTIONS[g]:
            jobs.append(dict(idx=i, genre=g, caption=cap, seed=BASE_SEED + i))
            i += 1
    return jobs[:50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "turbo"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    out_dir = os.path.join(OUT_ROOT, args.variant)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(OUT_ROOT, f"manifest_{args.variant}.csv")

    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            done = {int(r["idx"]) for r in csv.DictReader(f)}

    import torch
    h = AceStepHandler()
    status = h.initialize_service(
        project_root=ACESTEP_REPO,
        config_path=f"acestep-v15-{args.variant}",
        device="cpu",              # torch 侧全在 CPU(offload_dit_to_cpu 在 Mac 不生效,
        use_mlx_dit=True,          #  MPS 会被 fp32 主模型占满);扩散由 MLX 走 Metal,互不抢
    )
    print(status[0].splitlines()[0], flush=True)

    jobs = build_joblist()
    if args.limit:
        jobs = jobs[: args.limit]

    new_file = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=["idx", "genre", "caption", "seed", "variant", "path", "gen_s"])
        if new_file:
            w.writeheader()
        for job in jobs:
            if job["idx"] in done:
                continue
            t0 = time.time()
            params = GenerationParams(
                caption=job["caption"],
                duration=DURATION_S,
                thinking=False,
                instrumental=True,
                seed=job["seed"],
                # base 版标准流匹配采样要足步数+CFG;turbo 是 8 步蒸馏,用默认
                inference_steps=(30 if args.variant == "base" else 8),
            )
            config = GenerationConfig(batch_size=1, audio_format="flac", use_random_seed=False)
            result = generate_music(h, None, params, config, save_dir=out_dir)
            if not result.success:
                print(f"[{job['idx']}] FAIL: {result.error}")
                continue
            path = result.audios[0]["path"]
            dst = os.path.join(out_dir, f"as_{args.variant}_{job['idx']:03d}.flac")
            os.replace(path, dst)
            w.writerow({**job, "variant": args.variant, "path": dst, "gen_s": round(time.time() - t0, 1)})
            mf.flush()
            print(f"[{job['idx']}] {job['genre']:<10} {round(time.time()-t0,1)}s -> {os.path.basename(dst)}")
    print("DONE", args.variant)


if __name__ == "__main__":
    main()
