"""
跨生成器测试 · ACE-Step 1.5 批量生成 — 用 Suno 冻结 prompt 生成对照数据集。

从 suno_prompts_all.json 按 genre 均衡采样(固定种子,可复现),每条 prompt 生成一首
30s instrumental,输出 flac + manifest.csv。断点续跑:已生成的跳过。

在 ACE-Step-1.5 仓库目录下运行(依赖其 venv):
  cd ACE-Step-1.5
  uv run python crossgen_generate.py --prompts suno_prompts_all.json \
      --out output/acestep_batch --n-per-genre 125 --device cuda --backend vllm
  Mac 试跑:--device mps --backend mlx --lm acestep-5Hz-lm-0.6B --limit 2
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="suno_prompts_all.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-genre", type=int, default=125, help="125x8=1000")
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--config-path", default="acestep-v15-turbo")
    ap.add_argument("--lm", default="acestep-5Hz-lm-1.7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "pt", "mlx"])
    ap.add_argument("--offload", action="store_true", help="Mac/小显存:组件用完挪出显存")
    ap.add_argument("--limit", type=int, default=None, help="干跑:总共只生成 N 首")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"

    # ---- 均衡采样(固定种子,可复现) ----
    prompts = json.load(open(args.prompts))
    rng = np.random.default_rng(0)
    by_genre = {}
    for p in prompts:
        by_genre.setdefault(p["genre"], []).append(p)
    plan = []
    for g in sorted(by_genre):
        idx = rng.permutation(len(by_genre[g]))[: args.n_per_genre]
        for i in idx:
            p = by_genre[g][int(i)]
            plan.append(dict(audio_id=f"acestep_{g}_{int(i):04d}", genre=g,
                             caption=p["prompt"], seed=int(1e6 + i)))
    if args.limit:
        plan = plan[: args.limit]
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
    todo = [p for p in plan if p["audio_id"] not in done
            and not (out / f"{p['audio_id']}.flac").exists()]
    print(f"计划 {len(plan)} 首 | 已完成 {len(plan)-len(todo)} | 待生成 {len(todo)}")
    if not todo:
        return

    # ---- 初始化(一次) ----
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    t0 = time.time()
    dit = AceStepHandler()
    dit.initialize_service(project_root=os.getcwd(),
                           config_path=args.config_path, device=args.device,
                           offload_to_cpu=args.offload)
    llm = LLMHandler()
    llm.initialize(checkpoint_dir=args.checkpoint_dir,
                   lm_model_path=args.lm, backend=args.backend, device=args.device)
    print(f"模型加载完成({time.time()-t0:.0f}s)")

    # ---- 逐首生成 ----
    new_manifest = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.writer(mf)
    if new_manifest:
        w.writerow(["audio_id", "genre", "caption", "seed", "rel_path", "gen_time_s"])
    ok, fail, t0 = 0, 0, time.time()
    for i, p in enumerate(todo, 1):
        t1 = time.time()
        try:
            params = GenerationParams(
                caption=p["caption"], lyrics="[Instrumental]", instrumental=True,
                duration=args.duration, seed=p["seed"], thinking=True,
            )
            config = GenerationConfig(batch_size=1, audio_format="flac")
            r = generate_music(dit, llm, params, config, save_dir=str(out / "tmp"))
            if r.success and r.audios:
                src = Path(r.audios[0]["path"])
                dst = out / f"{p['audio_id']}.flac"
                src.replace(dst)
                w.writerow([p["audio_id"], p["genre"], p["caption"], p["seed"],
                            dst.name, round(time.time() - t1, 1)])
                mf.flush(); ok += 1
            else:
                print(f"  ❌ {p['audio_id']}: {r.error}"); fail += 1
        except Exception as e:
            print(f"  ❌ {p['audio_id']}: {e!r}"); fail += 1
        if i % 10 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            eta = rate * (len(todo) - i) / 60
            print(f"  {i}/{len(todo)} 完成 ({rate:.0f}s/首, 预计还需 {eta:.0f} 分钟, 失败 {fail})",
                  flush=True)
    mf.close()
    print(f"\n完成:成功 {ok},失败 {fail} -> {out}")


if __name__ == "__main__":
    main()
