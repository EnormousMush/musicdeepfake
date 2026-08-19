"""流匹配 2.0 · 零级:ACE-Step 自产语料 → common-spec 导出。

输入:gen_selfcorpus.py 的产物(Seagate 2_corpora_ai/acestep_self/{base,turbo}/ + manifest)。
输出:16k/mono/正中 10s/LUFS-23 FLAC(与 heldout_export 同链同规格),label=1、split=test。

Usage(Mac,repo .venv,librosa 依赖在里面):
  /Users/durunbao/Developer/frank-suno-backup/.venv/bin/python \
    part3_detector/flowmatch/selfcorpus_export.py
"""
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
IN_ROOT = Path(f"{SEAGATE}/2_corpora_ai/acestep_self")
OUT = Path(f"{SEAGATE}/3_workpacks/frank-suno-round1/flowmatch_export")
SPEC = dict(sr=16000, mono=True, crop_s=10.0, loudness_lufs=-23.0)


def main():
    man_path = OUT / "manifest.csv"
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
        print(f"resume: {len(done)} already exported", flush=True)
    new_file = not man_path.exists()
    mf = None
    n_ok, n_err, t0 = 0, 0, time.time()
    for variant in ("base", "turbo"):
        src_man = IN_ROOT / f"manifest_{variant}.csv"
        if not src_man.exists():
            print(f"skip {variant}: no manifest", flush=True)
            continue
        src_tag = f"acestep_{variant}"
        (OUT / "audio" / src_tag).mkdir(parents=True, exist_ok=True)
        if mf is None:
            mf = open(man_path, "a", newline="")
            w = csv.DictWriter(mf, fieldnames=["audio_id", "source", "label", "split",
                                               "rel_path", "genre", "seed"])
            if new_file:
                w.writeheader()
        for r in csv.DictReader(open(src_man)):
            aid = f"as_{variant}_{int(r['idx']):03d}"
            if aid in done:
                continue
            src = Path(r["path"])
            dst = OUT / "audio" / src_tag / f"{aid}.flac"
            try:
                dur = librosa.get_duration(path=str(src))
                pp = dict(SPEC, offset_s=max(0.0, (dur - SPEC["crop_s"]) / 2))
                y = load_canonical(str(src), pp)
                sf.write(dst, y, SPEC["sr"])
            except Exception as e:
                n_err += 1
                print(f"  {aid}: ERROR {repr(e)[:70]}", flush=True)
                continue
            w.writerow(dict(audio_id=aid, source=src_tag, label=1, split="test",
                            rel_path=f"audio/{src_tag}/{aid}.flac",
                            genre=r["genre"], seed=r["seed"]))
            mf.flush()
            n_ok += 1
    print(f"done: {n_ok} exported, {n_err} errors -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
