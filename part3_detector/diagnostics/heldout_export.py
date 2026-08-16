"""
第三来源 held-out 拼盘导出(2026-08-16,人类音乐集混淆 §6 第 7 发,终审前处理)。

输入:ccMixter 365 首 + IA netlabels 450 首(Seagate 1_corpora_real/,各自 manifest)。
输出:共同规格 10s FLAC(16k/mono/正中 10s/LUFS-23,与 jamendo_export 同链),
     全部 split=test、label=0 —— **只测不训**,这是终审协议的硬约束。
服务器侧放独立目录(data_store/heldout_export),不并入 crossgen_export 的 manifest
(避免污染 family_matrix 等按"非 fma 即生成器"逻辑的老脚本 —— jamendo 的教训)。

Usage(Mac,Seagate 挂载):
  python3 part3_detector/diagnostics/heldout_export.py \
    --out "/Volumes/Seagate /honors_paper/3_workpacks/frank-suno-round1/heldout_export"
  加 --limit 4 干跑。之后 rsync 到服务器 data_store/heldout_export/。
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
SOURCES = [
    ("ccmixter", f"{SEAGATE}/1_corpora_real/ccmixter", "manifest_ccmixter.csv"),
    ("ianet", f"{SEAGATE}/1_corpora_real/ia_netlabels", "manifest_ia.csv"),
]
SPEC = dict(sr=16000, mono=True, crop_s=10.0, loudness_lufs=-23.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    man_path = out / "manifest.csv"
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
        print(f"resume: {len(done)} already exported", flush=True)
    new_file = not man_path.exists()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    mf = open(man_path, "a", newline="")
    w = csv.DictWriter(mf, fieldnames=["audio_id", "source", "label", "split", "rel_path", "year"])
    if new_file:
        w.writeheader()

    n_ok, n_err, t0 = 0, 0, time.time()
    for src_tag, root, man in SOURCES:
        rows = list(csv.DictReader(open(Path(root) / man)))
        if args.limit:
            rows = rows[: args.limit]
        (out / "audio" / src_tag).mkdir(parents=True, exist_ok=True)
        for r in rows:
            aid = r["audio_id"]
            if aid in done:
                continue
            src = Path(root) / r["rel_path"]
            dst = out / "audio" / src_tag / f"{aid}.flac"
            try:
                dur = librosa.get_duration(path=str(src))
                pp = dict(SPEC, offset_s=max(0.0, (dur - SPEC["crop_s"]) / 2))
                y = load_canonical(str(src), pp)
                sf.write(dst, y, SPEC["sr"])
            except Exception as e:
                n_err += 1
                print(f"  {aid}: ERROR {repr(e)[:70]}", flush=True)
                continue
            w.writerow(dict(audio_id=aid, source=src_tag, label=0, split="test",
                            rel_path=f"audio/{src_tag}/{aid}.flac", year=r.get("year", "")))
            mf.flush()
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"  {n_ok} exported ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {n_ok} exported, {n_err} errors -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
