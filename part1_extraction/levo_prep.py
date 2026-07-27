"""
跨生成器测试 · LeVo 2(腾讯 SongGeneration v2)批量生成准备 + 结果收集。

LeVo 的 generate.sh 原生吃 jsonl 批量输入,所以不需要驱动循环——本脚本只负责:
  1) prep:从冻结 prompt 采样(与 ACE-Step/DiffRhythm2 同 1000 首)生成 lyrics.jsonl
  2) collect:扫描 LeVo 输出目录,按我们的命名/manifest 规范整理

用法(pod 上,LeVo 仓库根目录):
  # 准备输入
  python levo_prep.py prep --prompts suno_prompts_all.json --out levo_batch.jsonl
  # 生成(纯音乐模式;v2-large 22G 显存,不够加 --low_mem)
  sh generate.sh songgeneration_v2_large levo_batch.jsonl output_levo --bgm
  # 收集
  python levo_prep.py collect --jsonl levo_batch.jsonl \
      --levo-out output_levo --out levo_batch_1000

说明:
  - gt_lyric 用全器乐结构段([intro/inst/outro]),配合 --bgm 双保险出纯音乐;
    时长由结构决定(约 40-80s)。若干跑发现全器乐结构不被接受,备选方案:
    --with-lyrics 加占位歌词段(--bgm 下不会唱出来,只影响旋律结构)。
  - descriptions 默认用逐条冻结 prompt 原文(协议一致性优先);LeVo 官方要求
    逗号标签格式,若干跑风格跟随差,可用 --use-tags 换成 tags 字段对比。
  - LeVo 不暴露逐首 seed,manifest 中 seed 记为 -1(协议偏差,论文里注明)。
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

from crossgen_prompts import sample_plan

STRUCTURE_INST = "[intro-medium] ; [inst-medium] ; [inst-medium] ; [outro-medium]"
# 备选:带占位歌词(--bgm 下人声轨被丢弃,只用于撑起正常歌曲结构)
PLACEHOLDER_VERSE = ("The morning light is breaking through. The day begins anew. "
                     "Every step along the way. Leads to something true.")
PLACEHOLDER_CHORUS = ("We keep moving on and on. Through the night until the dawn. "
                      "Every moment carries song. This is where we all belong.")
STRUCTURE_LYRIC = (f"[intro-medium] ; [verse] {PLACEHOLDER_VERSE} ; "
                   f"[chorus] {PLACEHOLDER_CHORUS} ; [inst-medium] ; "
                   f"[chorus] {PLACEHOLDER_CHORUS} ; [outro-medium]")


def do_prep(args):
    plan = sample_plan(args.prompts, prefix="levo", n_per_genre=args.n_per_genre)
    if args.limit:
        plan = plan[: args.limit]
    gt_lyric = STRUCTURE_LYRIC if args.with_lyrics else STRUCTURE_INST
    with open(args.out, "w") as f:
        for p in plan:
            f.write(json.dumps({
                "idx": p["audio_id"],
                "gt_lyric": gt_lyric,
                "descriptions": p["tags"] if args.use_tags else p["caption"],
            }, ensure_ascii=False) + "\n")
    print(f"{len(plan)} 条 -> {args.out}(gt_lyric={'带占位歌词' if args.with_lyrics else '全器乐'}, "
          f"descriptions={'tags' if args.use_tags else '冻结prompt原文'})")


def do_collect(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = {json.loads(l)["idx"]: json.loads(l)
            for l in open(args.jsonl) if l.strip()}
    audio_dir = Path(args.levo_out)
    # LeVo 输出在 output_path/audio(s)/ 下,文件名 = idx;格式可能是 flac/wav/mp3
    files = {}
    for ext in ("flac", "wav", "mp3"):
        for p in audio_dir.rglob(f"*.{ext}"):
            files.setdefault(p.stem, p)
    with open(out / "manifest.csv", "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["audio_id", "genre", "caption", "seed", "rel_path", "gen_time_s"])
        ok = miss = 0
        for idx, row in plan.items():
            src = files.get(idx)
            if src is None:
                print(f"  缺失: {idx}"); miss += 1; continue
            dst = out / f"{idx}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
            genre = idx.split("_")[1] if "_" in idx else ""
            w.writerow([idx, genre, row["descriptions"], -1, dst.name, ""])
            ok += 1
    print(f"收齐 {ok},缺失 {miss} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prep")
    p1.add_argument("--prompts", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--n-per-genre", type=int, default=125)
    p1.add_argument("--limit", type=int, default=None, help="干跑:只取前 N 条")
    p1.add_argument("--with-lyrics", action="store_true",
                    help="gt_lyric 用占位歌词结构(备选方案)")
    p1.add_argument("--use-tags", action="store_true",
                    help="descriptions 用 tags 字段而非冻结 prompt 原文(备选方案)")
    p2 = sub.add_parser("collect")
    p2.add_argument("--jsonl", required=True)
    p2.add_argument("--levo-out", required=True)
    p2.add_argument("--out", required=True)
    args = ap.parse_args()
    do_prep(args) if args.cmd == "prep" else do_collect(args)


if __name__ == "__main__":
    main()
