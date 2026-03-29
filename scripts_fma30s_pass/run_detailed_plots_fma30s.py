"""
Sequential detailed_analysis runner for FMA jazz vocal 30s clips.
Generates both PNGs and a partial CSV per batch.

Usage:
    python run_detailed_plots_fma30s.py 0 20    # files 0–19
    python run_detailed_plots_fma30s.py 20 40   # files 20–39
    ... etc.
"""
import sys, os, csv, glob

ARTIFACTS_DIR = "/sessions/epic-youthful-goldberg/mnt/honors-project/Artifacts - Round 1"
MP3_DIR       = "/sessions/epic-youthful-goldberg/mnt/honors-project/Song_Gen/fma_jazz_vocal_30s"
OUTPUTS_DIR   = "/sessions/epic-youthful-goldberg/mnt/honors-project/fma_jazz_vocal_30s_results"
PLOTS_DIR     = os.path.join(OUTPUTS_DIR, "detailed_analysis_plots")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
sys.path.insert(0, ARTIFACTS_DIR)

import matplotlib
matplotlib.use("Agg")

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end   = int(sys.argv[2]) if len(sys.argv) > 2 else 20

all_files = sorted(glob.glob(os.path.join(MP3_DIR, "*.mp3")))
files = all_files[start:end]
print(f"Processing files {start}–{end-1} ({len(files)} files)…")

FIELDNAMES = [
    "filename", "tempo_bpm", "duration_s", "num_onsets", "num_beats",
    "rms_mean_db", "rms_std_db", "dynamic_range_db",
    "crest_mean", "crest_std", "zcr_mean", "spec_flat_mean",
    "centroid_mean_hz", "harm_perc_ratio", "error",
]

from detailed_analysis import analyze, plot_analysis

rows = []
for i, path in enumerate(files, 1):
    name = os.path.basename(path)
    global_idx = start + i
    print(f"  [{global_idx:3d}/{len(all_files)}] {name}", flush=True)
    try:
        data = analyze(path)
        stem = os.path.splitext(name)[0]
        png  = os.path.join(PLOTS_DIR, stem + ".png")
        plot_analysis(data, path, output_path=png)
        row = {"filename": name, "error": ""}
        row.update(data["stats"])
        print(f"    ✓ PNG saved: {stem}.png", flush=True)
    except Exception as e:
        row = {"filename": name, "error": str(e)}
        print(f"    ✗ Error: {e}", flush=True)
    rows.append(row)

out = os.path.join(OUTPUTS_DIR, f"detailed_part_{start}_{end}.csv")
for row in rows:
    for k in FIELDNAMES:
        row.setdefault(k, "")
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
print(f"\n  → Saved partial CSV: {out}")
