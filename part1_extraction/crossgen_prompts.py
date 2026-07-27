"""
跨生成器共享:冻结 prompt 均衡采样。

所有生成器批量脚本(acestep / diffrhythm2 / levo / diffrhythm1 / ...)都从这里拿
同一份 1000 首计划(seed 0,每 genre 125 条),保证跨生成器逐条同 prompt(内容持平)。
与 crossgen_generate.py 首批 ACE-Step 采样逐位一致(同 rng、同遍历顺序)。

单独运行可导出计划自查:
  python crossgen_prompts.py suno_prompts_all.json --prefix levo
"""
import json

import numpy as np


def sample_plan(prompts_path, prefix, n_per_genre=125):
    """返回 [{audio_id, genre, caption, tags, seed}, ...],顺序与首批 ACE-Step 完全一致。"""
    prompts = json.load(open(prompts_path))
    rng = np.random.default_rng(0)
    by_genre = {}
    for p in prompts:
        by_genre.setdefault(p["genre"], []).append(p)
    plan = []
    for g in sorted(by_genre):
        idx = rng.permutation(len(by_genre[g]))[:n_per_genre]
        for i in idx:
            p = by_genre[g][int(i)]
            plan.append(dict(audio_id=f"{prefix}_{g}_{int(i):04d}", genre=g,
                             caption=p["prompt"], tags=p.get("tags", ""),
                             seed=int(1e6 + i)))
    return plan


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--prefix", default="gen")
    ap.add_argument("--n-per-genre", type=int, default=125)
    args = ap.parse_args()
    plan = sample_plan(args.prompts, args.prefix, args.n_per_genre)
    print(f"{len(plan)} 首;前 3 条:")
    for p in plan[:3]:
        print(" ", p["audio_id"], "|", p["caption"][:60])
