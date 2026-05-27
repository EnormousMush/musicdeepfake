"""
Validation script: counts tracks, checks per-genre totals, flags anomalies.

Usage:
  python validate.py
  python validate.py --config path.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    manifest_path = Path(config["manifest_path"])
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    rows = load_manifest(manifest_path)
    audio_dir = Path(config["output_dir"])

    # --- Status breakdown ---
    status_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        status_counts[r["status"]] += 1

    print("=== Status breakdown ===")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:12s}: {count}")
    print(f"  {'TOTAL':12s}: {len(rows)}")

    # --- Per-genre track counts (2 tracks per done job) ---
    print("\n=== Per-genre track counts (done jobs × 2) ===")
    genre_done: dict[str, int] = defaultdict(int)
    genre_target: dict[str, int] = {
        g: gcfg["target_tracks"] for g, gcfg in config["genres"].items()
    }
    for r in rows:
        if r["status"] == "done":
            genre_done[r["genre"]] += 2

    all_ok = True
    for genre in sorted(config["genres"]):
        target = genre_target.get(genre, "?")
        got = genre_done.get(genre, 0)
        flag = "OK" if got >= target else f"UNDER by {target - got}"
        print(f"  {genre:12s}: {got:5d} / {target}  [{flag}]")
        if flag != "OK":
            all_ok = False

    # --- Duplicate ttapi_job_id check ---
    print("\n=== Duplicate ttapi_job_id check ===")
    seen_job_ids: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        jid = r.get("ttapi_job_id", "")
        if jid:
            seen_job_ids[jid].append(r["job_uid"])

    dupes = {jid: uids for jid, uids in seen_job_ids.items() if len(uids) > 1}
    if dupes:
        print(f"  WARN: {len(dupes)} duplicate ttapi_job_id(s) found:")
        for jid, uids in list(dupes.items())[:10]:
            print(f"    {jid}: {uids}")
    else:
        print("  No duplicates found.")

    # --- Missing audio files ---
    print("\n=== Missing audio files (done rows with no local path) ===")
    missing = [
        r for r in rows
        if r["status"] == "done" and not r.get("audio_path_1")
    ]
    if missing:
        print(f"  {len(missing)} done rows with no downloaded audio yet.")
    else:
        print("  All done rows have audio.")

    # --- Orphaned audio files (files on disk not in manifest) ---
    print("\n=== Orphaned audio files on disk ===")
    manifest_paths = set()
    for r in rows:
        for slot in ("1", "2"):
            p = r.get(f"audio_path_{slot}", "")
            if p:
                manifest_paths.add(Path(p))

    orphans = []
    if audio_dir.exists():
        for mp3 in audio_dir.rglob("*.mp3"):
            if mp3 not in manifest_paths:
                orphans.append(mp3)
    if orphans:
        print(f"  {len(orphans)} orphaned MP3(s) not in manifest:")
        for p in orphans[:10]:
            print(f"    {p}")
    else:
        print("  No orphaned files.")

    print("\n=== Summary ===")
    if all_ok and not dupes and not missing and not orphans:
        print("  All checks passed.")
    else:
        print("  Issues found — see above.")


if __name__ == "__main__":
    main()
