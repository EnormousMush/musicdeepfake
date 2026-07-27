"""
跨生成器测试 · Mac 端打包 — 把两边对齐到共同规格,导出自包含文件夹给服务器。

规格由 FakeMusicCaps 决定(它是 10s/16kHz):所有音频统一为 10s / 16kHz / mono / LUFS-23 / FLAC。
两边走完全相同的 load_canonical 处理链(防管线不对称自造混淆):
  - 我们的 round-1 clips(24k/30s FLAC):取中段 10s(offset 10s)重采样 16k 重 LUFS;
  - FakeMusicCaps(10s/16k wav):同一函数过一遍(offset 0)。
FakeMusicCaps 每个生成器随机采样 N 首(固定种子)。

输出 <out>/audio/<source>/<id>.flac + manifest.csv(audio_id, source, label, split, rel_path):
  - suno/fma 保留 round-1 的 train/val/test 划分(训练/域内评测用);
  - FakeMusicCaps 全部 split=test(纯评测,不参与训练)。

Usage (Mac, repo venv, Seagate 挂载):
  python part3_detector/diagnostics/crossgen_prep.py \
    --fmc-dir "/Volumes/Seagate /fakemusiccaps/FakeMusicCaps" \
    --round1-dir "/Volumes/Seagate /frank-suno-round1/subset_export_round1" \
    --out "/Volumes/Seagate /frank-suno-round1/crossgen_export" \
    --per-gen 1000
  加 --limit 10 干跑(每类/每生成器只处理 10 个)。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from part2_analysis.preprocess.audio import load_canonical

SPEC = dict(sr=16000, mono=True, crop_s=10.0, loudness_lufs=-23.0)
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def discover_generators(fmc_dir):
    """FakeMusicCaps 根目录下,每个含音频的子目录 = 一个生成器。"""
    gens = {}
    for d in sorted(Path(fmc_dir).iterdir()):
        if not d.is_dir() or d.name.startswith(("__", ".")):     # 跳过 __MACOSX 等元数据目录
            continue
        files = [p for p in d.rglob("*")
                 if p.suffix.lower() in AUDIO_EXT and not p.name.startswith("._")]
        if files:
            gens[d.name] = sorted(files)
    return gens


def process_one(src_path, out_path, offset_s):
    pp = dict(SPEC, offset_s=offset_s)
    y = load_canonical(str(src_path), pp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, y, SPEC["sr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmc-dir", required=True, help="解压后的 FakeMusicCaps 根目录")
    ap.add_argument("--round1-dir", default=None, help="round-1 导出目录;不给则只处理生成器侧")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-gen", type=int, default=1000)
    ap.add_argument("--gen-offset", type=float, default=0.0,
                    help="生成器侧裁剪起点秒(FMC 原始只有10s用0;自产30/40s歌用10取中段)")
    ap.add_argument("--limit", type=int, default=None, help="干跑:每组只处理 N 个")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows_out, failures = [], []
    t0 = time.time()

    # ---- 1) 我们的 round-1 clips -> 共同规格(中段 10s) ----
    r1_rows = []
    if args.round1_dir:
        r1 = Path(args.round1_dir)
        r1_rows = list(csv.DictReader(open(r1 / "manifest.csv")))
    for r in r1_rows:                                   # round-1 manifest 无 source 列,从 label 推
        r["source"] = "suno" if int(r["label"]) == 1 else "fma"
    if args.limit:
        keep, seen = [], {}
        for r in r1_rows:
            k = (r["source"], r["split"])
            if seen.get(k, 0) < args.limit:
                keep.append(r); seen[k] = seen.get(k, 0) + 1
        r1_rows = keep
    print(f"round-1 侧:{len(r1_rows)} clips -> 10s/16k")
    for i, r in enumerate(r1_rows, 1):
        rel = f"audio/{r['source']}/{r['audio_id']}.flac"
        try:
            if not (out / rel).exists():
                process_one(r1 / r["rel_path"], out / rel, offset_s=10.0)
            rows_out.append(dict(audio_id=r["audio_id"], source=r["source"],
                                 label=r["label"], split=r["split"], rel_path=rel))
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 500 == 0 or i == len(r1_rows):
            print(f"  {i}/{len(r1_rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)")

    # ---- 2) FakeMusicCaps -> 同一处理链 ----
    gens = discover_generators(args.fmc_dir)
    print(f"\nFakeMusicCaps 发现 {len(gens)} 个生成器目录: {list(gens)}")
    rng = np.random.default_rng(0)
    n = args.limit or args.per_gen
    for gname, files in gens.items():
        pick = list(rng.choice(files, min(n, len(files)), replace=False))
        print(f"  {gname}: {len(files)} 首,采样 {len(pick)}")
        for i, p in enumerate(pick, 1):
            aid = p.stem if p.stem.startswith(gname) else f"{gname}_{p.stem}"
            rel = f"audio/{gname}/{aid}.flac"
            try:
                if not (out / rel).exists():
                    process_one(p, out / rel, offset_s=args.gen_offset)
                rows_out.append(dict(audio_id=aid, source=gname,
                                     label=1, split="test", rel_path=rel))
            except Exception as e:
                failures.append((aid, repr(e)))
            if i % 500 == 0 or i == len(pick):
                print(f"    {i}/{len(pick)} ({time.time()-t0:.0f}s)")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_id", "source", "label", "split", "rel_path"])
        w.writeheader(); w.writerows(rows_out)
    print(f"\n完成:{len(rows_out)} clips -> {out}/manifest.csv  ({len(failures)} failed)")
    if failures:
        print(f"失败样例: {failures[:5]}")
    by = {}
    for r in rows_out:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print("各 source 数量:", by)


if __name__ == "__main__":
    main()
