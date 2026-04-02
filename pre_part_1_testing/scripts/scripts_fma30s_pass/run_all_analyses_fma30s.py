"""
Orchestration script: runs all analysis modules on all 104 FMA jazz vocal 30s clips
and saves one CSV per analysis type to the outputs folder.

Analyses:
  1. batch_bpm.py         → bpm_results.csv
  2. key.py               → key_results.csv
  3. lyrics_structure.py  → lyrics_structure_results.csv
  4. quantize_deg.py      → quantize_deg_results.csv
  5. instruments.py       → instruments_results.csv  (requires torch/transformers)
"""

import sys
import os
import csv
import glob
import traceback

# ── paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = "/sessions/epic-youthful-goldberg/mnt/honors-project/Artifacts - Round 1"
MP3_DIR       = "/sessions/epic-youthful-goldberg/mnt/honors-project/Song_Gen/fma_jazz_vocal_30s"
OUTPUTS_DIR   = "/sessions/epic-youthful-goldberg/mnt/honors-project/fma_jazz_vocal_30s_results"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

sys.path.insert(0, ARTIFACTS_DIR)

# ── collect mp3 files ─────────────────────────────────────────────────────────
mp3_files = sorted(glob.glob(os.path.join(MP3_DIR, "*.mp3")))
print(f"Found {len(mp3_files)} MP3 files.")


def short_name(path):
    return os.path.basename(path)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BPM  (uses batch_bpm.py — parallel multiprocessing, handles CSV itself)
# ─────────────────────────────────────────────────────────────────────────────
def run_bpm():
    from batch_bpm import batch_bpm
    out = os.path.join(OUTPUTS_DIR, "bpm_results.csv")
    batch_bpm(MP3_DIR, output_csv=out)
    print(f"  → Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# helpers for parallel runs
# ─────────────────────────────────────────────────────────────────────────────
from multiprocessing import Pool, cpu_count

N_WORKERS = min(cpu_count(), 8)


def _run_parallel(worker_fn, label, fieldnames, out_filename):
    out = os.path.join(OUTPUTS_DIR, out_filename)
    total = len(mp3_files)
    rows = []
    with Pool(processes=N_WORKERS) as pool:
        for i, row in enumerate(pool.imap(worker_fn, mp3_files), 1):
            status = "✗" if row.get("error") else "✓"
            print(f"  [{label}] [{i:3d}/{total}] {status}  {row['filename']}", flush=True)
            rows.append(row)
    for row in rows:
        for k in fieldnames:
            row.setdefault(k, "")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. KEY
# ─────────────────────────────────────────────────────────────────────────────
def _key_worker(path):
    sys.path.insert(0, ARTIFACTS_DIR)
    from key import estimate_key
    try:
        best_key, best_corr, alt_key, alt_corr = estimate_key(path)
        return {
            "filename": short_name(path),
            "best_key": best_key,
            "best_corr": best_corr,
            "alt_key": alt_key or "",
            "alt_corr": alt_corr if alt_corr is not None else "",
            "error": "",
        }
    except Exception as e:
        return {"filename": short_name(path), "best_key": "", "best_corr": "",
                "alt_key": "", "alt_corr": "", "error": str(e)}


def run_key():
    _run_parallel(
        _key_worker, "key",
        ["filename", "best_key", "best_corr", "alt_key", "alt_corr", "error"],
        "key_results.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LYRICS / SONG STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
def _structure_worker(path):
    import json
    sys.path.insert(0, ARTIFACTS_DIR)
    from lyrics_structure import analyze_structure
    try:
        sections, R, _ = analyze_structure(path)
        if not sections:
            return {
                "filename": short_name(path),
                "num_sections": 0,
                "sections_json": "[]",
                "section_names": "",
                "error": "no structure detected",
            }
        section_names = " | ".join(
            f"{s['name']}({s['letter']}) {s['start']:.1f}-{s['end']:.1f}s"
            for s in sections
        )
        return {
            "filename": short_name(path),
            "num_sections": len(sections),
            "sections_json": json.dumps(sections),
            "section_names": section_names,
            "error": "",
        }
    except Exception as e:
        return {
            "filename": short_name(path),
            "num_sections": "",
            "sections_json": "",
            "section_names": "",
            "error": str(e),
        }


def run_lyrics_structure():
    _run_parallel(
        _structure_worker, "structure",
        ["filename", "num_sections", "section_names", "sections_json", "error"],
        "lyrics_structure_results.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. QUANTIZATION DEGREE
# ─────────────────────────────────────────────────────────────────────────────
def _quantize_worker(path):
    sys.path.insert(0, ARTIFACTS_DIR)
    from quantize_deg import analyze_quantization
    try:
        result = analyze_quantization(path)
        if "error" in result:
            return {"filename": short_name(path), "error": result["error"]}
        row = {"filename": short_name(path), "error": ""}
        row.update(result)
        return row
    except Exception as e:
        return {"filename": short_name(path), "error": str(e)}


def run_quantize_deg():
    _run_parallel(
        _quantize_worker, "quantize",
        [
            "filename",
            "tempo", "quantization_score", "mean_dev_ms", "std_dev_ms", "max_dev_ms",
            "swing_pct", "num_onsets", "subdivision",
            "error",
        ],
        "quantize_deg_results.csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. INSTRUMENTS (optional – requires torch + transformers)
# ─────────────────────────────────────────────────────────────────────────────
def run_instruments():
    try:
        from instruments import detect_instruments
    except ImportError as e:
        print(f"  [instruments] Skipped — missing dependency: {e}")
        return

    rows = []
    for i, path in enumerate(mp3_files, 1):
        print(f"  [instruments] {i}/{len(mp3_files)} {short_name(path)}", flush=True)
        try:
            detected = detect_instruments(path)
            row = {"filename": short_name(path), "error": ""}
            for rank, (label, score) in enumerate(detected, 1):
                row[f"instrument_{rank}"] = label
                row[f"score_{rank}"] = round(score, 4)
            rows.append(row)
        except Exception as e:
            rows.append({"filename": short_name(path), "error": str(e)})

    if not rows:
        return

    max_rank = max(
        sum(1 for k in r if k.startswith("instrument_")) for r in rows
    )
    fieldnames = ["filename"]
    for r in range(1, max_rank + 1):
        fieldnames += [f"instrument_{r}", f"score_{r}"]
    fieldnames.append("error")

    for row in rows:
        for k in fieldnames:
            row.setdefault(k, "")

    out = os.path.join(OUTPUTS_DIR, "instruments_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analyses", nargs="+",
        choices=["bpm", "key", "structure", "quantize", "instruments", "all"],
        default=["all"],
        help="Which analyses to run (default: all)"
    )
    args = parser.parse_args()

    run_all = "all" in args.analyses
    analyses = args.analyses

    if run_all or "bpm" in analyses:
        print("\n=== BPM ===")
        run_bpm()

    if run_all or "key" in analyses:
        print("\n=== KEY ===")
        run_key()

    if run_all or "structure" in analyses:
        print("\n=== SONG STRUCTURE ===")
        run_lyrics_structure()

    if run_all or "quantize" in analyses:
        print("\n=== QUANTIZATION ===")
        run_quantize_deg()

    if run_all or "instruments" in analyses:
        print("\n=== INSTRUMENTS (CLAP) ===")
        run_instruments()

    print("\n✓ All done. CSVs saved to:", OUTPUTS_DIR)
