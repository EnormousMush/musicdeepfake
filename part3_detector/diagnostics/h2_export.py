"""
H2a 导出:三方 30 秒共同窗 + suno 正中 10 秒(采样位置混淆裁决),Mac 端打包。

背景:H1 的"歌内活跃"(std 类特征 suno 偏高)带采样位置混淆——suno 旧切片偏早
(可能撞 build-up),fma/jamendo 取中段。本导出把三方对齐到可比窗口:
  suno30 / jam30:全长原曲取正中 30s;fma30:其 30s 摘录原样过链(offset 0);
  suno10mid:全长取正中 10s —— 与 jamendo 现有中段 10s 直接对表,位置混淆即裁。
全部 16k/mono/LUFS-23/FLAC(与共同规格同链 load_canonical)。

Usage (Mac, repo venv, Seagate 挂载;全量约 9000 个解码,预计 1h):
  python part3_detector/diagnostics/h2_export.py \
    --out "/Volumes/Seagate /honors_paper/3_workpacks/frank-suno-round1/h2_export"
  加 --limit 10 干跑。产物 rsync 服务器后用 extract_features(30s 版)提特征。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import librosa
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from part2_analysis.preprocess.audio import load_canonical

SEAGATE = "/Volumes/Seagate /honors_paper"
CROSSGEN_MAN = f"{SEAGATE}/3_workpacks/frank-suno-round1/crossgen_export/manifest.csv"
ROUND1_MAN = f"{SEAGATE}/3_workpacks/frank-suno-round1/subset_export_round1/manifest.csv"
ROUND1_AUDIO = f"{SEAGATE}/3_workpacks/frank-suno-round1/subset_export_round1"
SUNO_FULL = f"{SEAGATE}/2_corpora_ai/suno_audio"
JAM_MAN = f"{SEAGATE}/1_corpora_real/jamendo2025/manifest_jamendo.csv"


def mid_offset(path, crop_s):
    dur = librosa.get_duration(path=str(path))
    return max(0.0, (dur - crop_s) / 2)


def export_one(src, dst, crop_s, offset_s):
    pp = dict(sr=16000, mono=True, crop_s=crop_s, loudness_lufs=-23.0, offset_s=offset_s)
    y = load_canonical(str(src), pp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, 16000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- 任务清单 ---
    # suno:crossgen 3000 条 -> 全长 mp3(uid_slot 命名,genre 来自 round1 manifest)
    genre_of = {r["audio_id"]: r["genre"] for r in csv.DictReader(open(ROUND1_MAN))
                if r["audio_id"].startswith("suno_")}
    # round1 genre 字段:suno 行是简单词(pop/rock…)
    cg = list(csv.DictReader(open(CROSSGEN_MAN)))
    jobs = []   # (audio_id, source_tag, src_path, crop_s, offset_mode, split)
    for r in cg:
        if r["source"] == "suno":
            uid_slot = r["audio_id"][len("suno_"):]
            g = genre_of.get(r["audio_id"], "")
            src = Path(SUNO_FULL) / g / f"{uid_slot}.mp3"
            jobs.append((r["audio_id"], "suno30", src, 30.0, "mid", r["split"]))
            jobs.append((r["audio_id"], "suno10mid", src, 10.0, "mid", r["split"]))
        elif r["source"] == "fma":
            src = Path(ROUND1_AUDIO) / f"audio/{r['audio_id']}.flac"
            jobs.append((r["audio_id"], "fma30", src, 30.0, "zero", r["split"]))
    for r in csv.DictReader(open(JAM_MAN)):
        jobs.append((r["audio_id"], "jam30", Path(r["audio_path"]), 30.0, "mid", ""))
    # jamendo split 从 crossgen_add_jamendo manifest 补
    jam_split = {}
    jman = Path(f"{SEAGATE}/3_workpacks/frank-suno-round1/crossgen_add_jamendo/jamendo_manifest.csv")
    if jman.exists():
        jam_split = {r["audio_id"]: r["split"] for r in csv.DictReader(open(jman))}
    if args.limit:
        by_tag = {}
        keep = []
        for j in jobs:
            if by_tag.get(j[1], 0) < args.limit:
                keep.append(j); by_tag[j[1]] = by_tag.get(j[1], 0) + 1
        jobs = keep
    print(f"{len(jobs)} export jobs", flush=True)

    man_path = out / "h2_manifest.csv"
    done = set()
    if man_path.exists():
        done = {r["audio_id"] + "|" + r["window"] for r in csv.DictReader(open(man_path))}
        print(f"resume: {len(done)}")
    new_file = not man_path.exists()
    t0, n = time.time(), 0
    with open(man_path, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=["audio_id", "window", "source", "label",
                                           "split", "rel_path"])
        if new_file:
            w.writeheader()
        # 去重(jamendo manifest 双 genre 行)
        seen_pairs = set()
        for aid, tag, src, crop, om, split in jobs:
            if (aid, tag) in seen_pairs:
                continue
            seen_pairs.add((aid, tag))
            if f"{aid}|{tag}" in done:
                continue
            dst = out / "audio" / tag / f"{aid}.flac"
            try:
                off = mid_offset(src, crop) if om == "mid" else 0.0
                export_one(src, dst, crop, off)
            except Exception as e:
                print(f"  {tag}/{aid}: ERROR {repr(e)[:70]}", flush=True)
                continue
            base = tag.rstrip("0123456789").rstrip("_")
            source = {"suno30": "suno", "suno10mid": "suno",
                      "fma30": "fma", "jam30": "jamendo"}[tag]
            sp = split or jam_split.get(aid, "")
            w.writerow(dict(audio_id=aid, window=tag, source=source,
                            label=1 if source == "suno" else 0, split=sp,
                            rel_path=f"audio/{tag}/{aid}.flac"))
            mf.flush()
            n += 1
            if n % 200 == 0:
                print(f"  {n} exported ({time.time()-t0:.0f}s)", flush=True)
    print(f"done -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
