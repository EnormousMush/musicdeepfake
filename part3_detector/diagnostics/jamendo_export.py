"""
Jamendo 2024–2026 现代人类语料 → 共同规格导出(年代混淆拆旗,Mac 端打包)。

与 crossgen_prep 同链(load_canonical:16k/mono/10s/LUFS-23/FLAC),三个关键设计:
  - 取每首歌**正中间** 10s(offset=(dur-10)/2),对齐 FMA 的"歌曲中段"采样方式,
    避免把"截取位置"混进年代对照;
  - 按 track_id 去重(57 首双 genre 歌只保留首次出现,防双倍加权);
  - train/val/test = 70/15/15 **按 artist 切**(同一歌手不跨集,防泄漏)。

Usage (Mac, repo venv, Seagate 挂载):
  python part3_detector/diagnostics/jamendo_export.py \
    --raw-dir "/Volumes/Seagate /honors_paper/1_corpora_real/jamendo2025" \
    --out "/Volumes/Seagate /honors_paper/3_workpacks/frank-suno-round1/crossgen_add_jamendo"
  加 --limit 20 干跑。之后 rsync audio/ 到服务器 crossgen_export/audio/,合并 manifest(见 runbook)。
"""
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import librosa
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from part2_analysis.preprocess.audio import load_canonical

SPEC = dict(sr=16000, mono=True, crop_s=10.0, loudness_lufs=-23.0)


def artist_split(artist_id: str) -> str:
    """歌手级确定性划分:md5 哈希 → 70/15/15。"""
    h = int(hashlib.md5(str(artist_id).encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out)
    (out / "audio" / "jamendo").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(raw / "manifest_jamendo.csv")))
    seen_tracks, todo = set(), []
    for r in rows:
        if r["track_id"] in seen_tracks:
            continue                      # 双 genre 重复,保留首次
        seen_tracks.add(r["track_id"])
        todo.append(r)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} rows -> {len(todo)} unique tracks", flush=True)

    man_path = out / "jamendo_manifest.csv"
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
        print(f"resume: {len(done)} already exported", flush=True)
    new_file = not man_path.exists()
    t0 = time.time()
    with open(man_path, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=["audio_id", "source", "label", "split", "rel_path"])
        if new_file:
            w.writeheader()
        n_done = 0
        for r in todo:
            aid = r["audio_id"]
            if aid in done:
                continue
            src = raw / r["genre"] / f"{aid}.mp3"
            dst = out / "audio" / "jamendo" / f"{aid}.flac"
            try:
                dur = librosa.get_duration(path=str(src))
                pp = dict(SPEC, offset_s=max(0.0, (dur - SPEC["crop_s"]) / 2))
                y = load_canonical(str(src), pp)
                sf.write(dst, y, SPEC["sr"])
            except Exception as e:
                print(f"  {aid}: ERROR {repr(e)[:70]}", flush=True)
                continue
            w.writerow(dict(audio_id=aid, source="jamendo", label=0,
                            split=artist_split(r["artist_id"]),
                            rel_path=f"audio/jamendo/{aid}.flac"))
            mf.flush()
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done} exported ({time.time()-t0:.0f}s)", flush=True)
    print(f"done -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
