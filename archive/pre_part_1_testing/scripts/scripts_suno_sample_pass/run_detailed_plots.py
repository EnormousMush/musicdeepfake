"""
Runs detailed_analysis on a slice of MP3 files, generating both PNGs and a partial CSV.
Usage:  python run_detailed_plots.py 0 50    # files 0..49
        python run_detailed_plots.py 50 100  # files 50..99
"""
import sys, os, csv, glob
from multiprocessing import Pool, cpu_count

ARTIFACTS_DIR = "/sessions/epic-youthful-goldberg/mnt/honors-project/Artifacts - Round 1"
MP3_DIR       = "/sessions/epic-youthful-goldberg/mnt/honors-project/Song_Gen/outputs_jazz_test"
OUTPUTS_DIR   = "/sessions/epic-youthful-goldberg/mnt/outputs"
PLOTS_DIR     = os.path.join(OUTPUTS_DIR, "detailed_analysis_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
sys.path.insert(0, ARTIFACTS_DIR)

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end   = int(sys.argv[2]) if len(sys.argv) > 2 else 50

all_files = sorted(glob.glob(os.path.join(MP3_DIR, "*.mp3")))
files = all_files[start:end]
print(f"Processing files {start}–{end-1} ({len(files)} files)…")

FIELDNAMES = [
    "filename", "tempo_bpm", "duration_s", "num_onsets", "num_beats",
    "rms_mean_db", "rms_std_db", "dynamic_range_db",
    "crest_mean", "crest_std", "zcr_mean", "spec_flat_mean",
    "centroid_mean_hz", "harm_perc_ratio", "error",
]

def worker(path):
    sys.path.insert(0, ARTIFACTS_DIR)
    from detailed_analysis import analyze, plot_analysis
    name = os.path.basename(path)
    try:
        data = analyze(path)
        stem = os.path.splitext(name)[0]
        png  = os.path.join(PLOTS_DIR, stem + ".png")
        plot_analysis(data, path, output_path=png)
        row = {"filename": name, "error": ""}
        row.update(data["stats"])
        return row
    except Exception as e:
        return {"filename": name, "error": str(e)}

N = min(cpu_count(), 4)
rows = []
with Pool(processes=N) as pool:
    for i, row in enumerate(pool.imap(worker, files), start + 1):
        status = "✗" if row.get("error") else "✓"
        print(f"  [{i:3d}/{len(all_files)}] {status}  {row['filename']}", flush=True)
        rows.append(row)

# Write partial CSV (will be merged later)
out = os.path.join(OUTPUTS_DIR, f"detailed_part_{start}_{end}.csv")
for row in rows:
    for k in FIELDNAMES:
        row.setdefault(k, "")
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
print(f"  → Saved partial CSV: {out}")
