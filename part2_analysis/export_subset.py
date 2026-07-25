"""
Export the Stage-0 subset as a portable, preprocessed folder to rsync to the GPU server.

Each of the 4000 subset tracks is decoded -> mono -> resampled (24 kHz, MERT's rate)
-> cropped (per config) -> LUFS-normalized, then written as FLAC. The canonical
preprocessing (confound neutralization) happens here on the Mac, once; the server
only runs the encoder. Output folder is self-contained (audio + manifest).

Usage:
  python export_subset.py --config configs/stage0_mel_linear.yaml            # all 4000
  python export_subset.py --config configs/stage0_mel_linear.yaml --limit 10 # dry-run
"""
import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # part2_analysis (manifest + preprocess, both Part 2)
import manifest as M
from preprocess.audio import load_canonical

EXPORT_SR = 24000  # MERT sampling rate; wav2vec2 (16 kHz) is resampled server-side


def _subsample(man, limit):
    out, seen = [], {0: 0, 1: 0}
    for r in man:
        if seen[r["label"]] < limit:
            out.append(r); seen[r["label"]] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data_store/subset_export")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pp = dict(cfg["preprocess"]); pp["sr"] = EXPORT_SR   # export at 24 kHz

    man = M.load_manifest(cfg["paths"]["manifest"])
    if args.limit:
        man = _subsample(man, args.limit)

    outdir = Path(args.out)
    (outdir / "audio").mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for i, r in enumerate(man, 1):
        rel = f"audio/{r['audio_id']}.flac"
        dst = outdir / rel
        if not dst.exists():
            try:
                y = load_canonical(r["path"], pp)
                sf.write(dst, y, EXPORT_SR, format="FLAC")
            except Exception as e:
                failures.append((r["audio_id"], repr(e)))
                continue
        rows.append({"audio_id": r["audio_id"], "rel_path": rel, "label": r["label"],
                     "group_id": r["group_id"], "genre": r["genre"], "split": r["split"]})
        if i % 200 == 0 or i == len(man):
            print(f"  {i}/{len(man)} exported ({len(failures)} failed)")

    with open(outdir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_id", "rel_path", "label", "group_id", "genre", "split"])
        w.writeheader(); w.writerows(rows)

    sr_note = outdir / "README.txt"
    sr_note.write_text(
        f"Preprocessed subset for Stage 1 (server).\n"
        f"  audio/: {len(rows)} FLAC clips, mono, {EXPORT_SR} Hz, "
        f"{pp['crop_s']}s from offset {pp['offset_s']}s, LUFS {pp['loudness_lufs']}.\n"
        f"  manifest.csv: audio_id, rel_path, label(1=AI/0=human), group_id, genre, split.\n"
        f"Run run_stage1.py on the server pointing --data-dir here.\n")

    total_mb = sum(p.stat().st_size for p in (outdir / "audio").glob("*.flac")) / 1e6
    print(f"\nExported {len(rows)} clips ({total_mb:.0f} MB) -> {outdir}")
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")


if __name__ == "__main__":
    main()
