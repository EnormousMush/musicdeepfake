"""
Batch BPM Estimator
===================
Processes a folder of audio files in parallel (one worker per CPU core).
Much faster than running bpm.py once per file from the shell.

Usage:
    python batch_bpm.py /path/to/songs/
    python batch_bpm.py /path/to/songs/ --output results.csv --workers 4
"""

import argparse
import csv
import os
import sys
import traceback
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Import estimate_bpm from bpm.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from bpm import estimate_bpm

_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff"}


def _process(audio_path: str):
    """Worker function — runs in a separate process."""
    try:
        bpm = estimate_bpm(audio_path)
        return (Path(audio_path).name, round(bpm, 2), None)
    except Exception as e:
        return (Path(audio_path).name, None, str(e))


def batch_bpm(folder: str, output_csv: str = "bpm_results.csv", workers: int = None):
    files = sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in _AUDIO_EXTS
    )

    if not files:
        print(f"No audio files found in: {folder}")
        return

    n_workers = workers or min(cpu_count(), len(files))
    print(f"Processing {len(files)} files with {n_workers} parallel workers...\n")

    results = []
    with Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap(_process, files), 1):
            name, bpm, err = result
            if err:
                print(f"[{i:3d}/{len(files)}] ✗  {name}  — {err}")
            else:
                print(f"[{i:3d}/{len(files)}] ✓  {name:<50} {bpm} BPM")
            results.append(result)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "bpm", "error"])
        for name, bpm, err in results:
            writer.writerow([name, bpm or "", err or ""])

    success = sum(1 for _, bpm, _ in results if bpm is not None)
    print(f"\nDone: {success}/{len(files)} succeeded → {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Batch BPM estimation with parallel processing.")
    parser.add_argument("folder", help="Folder containing audio files")
    parser.add_argument("--output", "-o", default="bpm_results.csv",
                        help="Output CSV path (default: bpm_results.csv)")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of parallel workers (default: number of CPU cores)")
    args = parser.parse_args()

    batch_bpm(args.folder, output_csv=args.output, workers=args.workers)


if __name__ == "__main__":
    main()
