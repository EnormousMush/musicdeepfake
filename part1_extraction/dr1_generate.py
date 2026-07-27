"""
跨生成器测试 · DiffRhythm 1(v1.2)批量生成 — 同一份冻结 prompt,1000 首 instrumental。

DiffRhythm v1 官方 infer.py 一次只生成一首且每次重载模型,这里模型加载一次后循环。
instrumental 做法:空 lrc + style prompt 后缀 ", instrumental, no vocals"
(v1 没有 DiffRhythm2 的结构标签机制,靠 MuLan 文本风格约束;干跑 10 首先验人声)。

在 DiffRhythm 仓库根目录运行(依赖其环境;权重首跑自动从 HF 拉 ASLP-lab/DiffRhythm-1_2):
  python dr1_generate.py --prompts suno_prompts_all.json --out output/dr1_batch --chunked
  干跑:--limit 10
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "infer")

import torch
import torchaudio
from einops import rearrange

from crossgen_prompts import sample_plan
from infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)


# 与官方 infer/infer.py 的 inference() 一致(steps=32, cfg=4.0),仅精简为单首返回
def run_one(cfm, vae, cond, text, duration, style_prompt, negative_style_prompt,
            start_time, pred_frames, song_duration, chunked):
    with torch.inference_mode():
        latents, _ = cfm.sample(
            cond=cond, text=text, duration=duration, style_prompt=style_prompt,
            max_duration=duration, song_duration=song_duration,
            negative_style_prompt=negative_style_prompt, steps=32, cfg_strength=4.0,
            start_time=start_time, latent_pred_segments=pred_frames, batch_infer_num=1,
        )
        latent = latents[0].to(torch.float32).transpose(1, 2)
        output = decode_audio(latent, vae, chunked=chunked)
        output = rearrange(output, "b d n -> d (b n)")
        return (output.to(torch.float32).div(torch.max(torch.abs(output)))
                .clamp(-1, 1).mul(32767).to(torch.int16).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="suno_prompts_all.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-genre", type=int, default=125)
    ap.add_argument("--audio-length", type=int, default=95, help="95 -> max_frames 2048")
    ap.add_argument("--style-suffix", default=", instrumental, no vocals")
    ap.add_argument("--chunked", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="干跑:总共只生成 N 首")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"

    plan = sample_plan(args.prompts, prefix="dr1", n_per_genre=args.n_per_genre)
    if args.limit:
        plan = plan[: args.limit]
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
    todo = [p for p in plan if p["audio_id"] not in done
            and not (out / f"{p['audio_id']}.flac").exists()]
    print(f"计划 {len(plan)} 首 | 已完成 {len(plan)-len(todo)} | 待生成 {len(todo)}", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_frames = 2048 if args.audio_length == 95 else 6144
    t0 = time.time()
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    print(f"模型加载完成({time.time()-t0:.0f}s)", flush=True)

    # 逐首不变的条件只算一次:空 lrc(instrumental)+ 无编辑参考
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, "", tokenizer, args.audio_length, device)
    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, False, None, None, vae)

    new_manifest = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.writer(mf)
    if new_manifest:
        w.writerow(["audio_id", "genre", "caption", "seed", "rel_path", "gen_time_s"])
    ok, fail, t0 = 0, 0, time.time()
    for i, p in enumerate(todo, 1):
        t1 = time.time()
        try:
            torch.manual_seed(p["seed"])
            style_prompt = get_style_prompt(muq, prompt=p["caption"] + args.style_suffix)
            audio = run_one(cfm, vae, latent_prompt, lrc_prompt, end_frame,
                            style_prompt, negative_style_prompt, start_time,
                            pred_frames, song_duration, args.chunked)
            dst = out / f"{p['audio_id']}.flac"
            try:
                torchaudio.save(str(dst), audio, sample_rate=44100)
            except Exception:
                dst = out / f"{p['audio_id']}.wav"
                torchaudio.save(str(dst), audio, sample_rate=44100)
            w.writerow([p["audio_id"], p["genre"], p["caption"], p["seed"],
                        dst.name, round(time.time() - t1, 1)])
            mf.flush(); ok += 1
        except Exception as e:
            print(f"  ❌ {p['audio_id']}: {e!r}", flush=True); fail += 1
        if i % 10 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            eta = rate * (len(todo) - i) / 60
            print(f"  {i}/{len(todo)} 完成 ({rate:.0f}s/首, 预计还需 {eta:.0f} 分钟, 失败 {fail})",
                  flush=True)
    mf.close()
    print(f"\n完成:成功 {ok},失败 {fail} -> {out}")


if __name__ == "__main__":
    main()
