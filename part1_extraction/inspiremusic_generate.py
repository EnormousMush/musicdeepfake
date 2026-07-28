"""
跨生成器测试 · InspireMusic(阿里 FunAudioLLM)批量生成 — 同冻结 prompt 1000 首。

LM+流匹配家族第 4 员(Qwen2.5 AR + flow matching 渲染),原生纯器乐(无人声机制)。
模型加载一次循环生成;断点续跑 manifest 同老款。

在 InspireMusic 仓库根目录运行(依赖其环境;模型先 git clone 到 pretrained_models/):
  python inspiremusic_generate.py --prompts suno_prompts_all.json \
      --out output/inspire_batch --model InspireMusic-1.5B-Long
  干跑:--limit 10
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from crossgen_prompts import sample_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="suno_prompts_all.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="InspireMusic-1.5B-Long")
    ap.add_argument("--model-dir", default=None, help="默认 pretrained_models/<model>")
    ap.add_argument("--n-per-genre", type=int, default=125)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--chorus", default="intro", help="结构标签: intro/verse/chorus/outro")
    ap.add_argument("--fast", action="store_true", help="跳过流匹配(降质提速,不建议)")
    ap.add_argument("--format", default="wav", help="输出格式(flac 若其保存器支持)")
    ap.add_argument("--limit", type=int, default=None, help="干跑:总共只生成 N 首")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"

    plan = sample_plan(args.prompts, prefix="inspire", n_per_genre=args.n_per_genre)
    if args.limit:
        plan = plan[: args.limit]
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
    todo = [p for p in plan if p["audio_id"] not in done
            and not (out / f"{p['audio_id']}.{args.format}").exists()]
    print(f"计划 {len(plan)} 首 | 已完成 {len(plan)-len(todo)} | 待生成 {len(todo)}", flush=True)
    if not todo:
        return

    from inspiremusic.cli.inference import InspireMusicModel, env_variables
    env_variables()
    t0 = time.time()
    model = InspireMusicModel(
        model_name=args.model,
        model_dir=args.model_dir,
        min_generate_audio_seconds=10.0,
        max_generate_audio_seconds=args.duration,
        gpu=0,
        result_dir=str(out),
        fast=args.fast,
    )
    print(f"模型加载完成({time.time()-t0:.0f}s)", flush=True)

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
            model.inference(
                task="text-to-music",
                text=p["caption"],
                chorus=args.chorus,
                time_start=0.0,
                time_end=args.duration,
                output_fn=p["audio_id"],
                output_format=args.format,
            )
            dst = out / f"{p['audio_id']}.{args.format}"
            if not dst.exists():
                raise RuntimeError(f"未产出 {dst.name}")
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
