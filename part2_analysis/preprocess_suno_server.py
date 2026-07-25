"""
Server-side: generate the Suno half of the subset from the raw mp3s already on the
server, so we don't re-upload them. Uses the SAME canonical preprocessing as the FMA
clips that were exported on the Mac (mono, 24 kHz, 30 s crop, LUFS) — identical code path.

For each label==1 (Suno) row in the manifest, reconstruct the raw path
  <suno-root>/<genre>/<uuid>_<n>.mp3   (audio_id = "suno_<uuid>_<n>")
preprocess it, and write audio/<audio_id>.flac into the export dir. FMA clips arrive
via rsync from the Mac; Suno clips are made here. Missing tracks are skipped.

Usage (on the server, in the detector dir, venv active):
  python preprocess_suno_server.py --data-dir data_store/subset_export
"""
import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import soundfile as sf
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess.audio import load_canonical

EXPORT_SR = 24000  # must match export_subset.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="subset_export dir (has manifest.csv, audio/)")
    ap.add_argument("--suno-root", default="/mindata/frank-suno/part1_extraction/audio")
    ap.add_argument("--config", default="configs/stage0_mel_linear.yaml")
    args = ap.parse_args()

    pp = dict(yaml.safe_load(open(args.config))["preprocess"])
    pp["sr"] = EXPORT_SR

    data_dir = Path(args.data_dir)
    (data_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    suno = [r for r in rows if int(r["label"]) == 1]
    print(f"Suno rows in manifest: {len(suno)}; source: {args.suno_root}")

    done, skipped, missing, failed = 0, 0, 0, []
    for i, r in enumerate(suno, 1):
        dst = data_dir / r["rel_path"]              # audio/suno_<uuid>_<n>.flac
        if dst.exists():
            skipped += 1
            continue
        base = r["audio_id"][len("suno_"):]         # <uuid>_<n>
        src = Path(args.suno_root) / r["genre"] / f"{base}.mp3"
        if not src.exists():
            missing += 1
            continue
        try:
            y = load_canonical(str(src), pp)
            sf.write(dst, y, EXPORT_SR, format="FLAC")
            done += 1
        except Exception as e:
            failed.append((r["audio_id"], repr(e)))
        if i % 200 == 0 or i == len(suno):
            print(f"  {i}/{len(suno)}  (made {done}, cached {skipped}, missing {missing}, failed {len(failed)})")

    print(f"\nDone: {done} made, {skipped} already there, {missing} not on server, {len(failed)} failed.")
    if missing:
        print("  (missing = the ~500 tracks on the Mac but not the server; safe to ignore.)")
    if failed:
        print(f"  first failures: {failed[:3]}")
    have = len(list((data_dir / 'audio').glob('suno_*.flac')))
    print(f"Suno FLACs now present: {have}")


if __name__ == "__main__":
    main()
