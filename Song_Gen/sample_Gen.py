import os
import time
import requests
from pathlib import Path

BASE_URL = "https://api.ttapi.io/suno/v1/music"
API_KEY = os.environ.get("TT_API_KEY")

if not API_KEY:
    raise RuntimeError("TT_API_KEY environment variable is not set")

HEADERS = {
    "TT-API-KEY": API_KEY,
    "Content-Type": "application/json",
}

OUTPUT_DIR = Path("outputs_jazz_test")
OUTPUT_DIR.mkdir(exist_ok=True)


def submit_inspiration_job(max_retries: int = 5):
    """
    Inspiration mode (custom=false), jazz vibe prompt.
    Each call returns job_id and will generate 2 tracks.
    Retries on 429/499 rate-limit responses with exponential backoff.
    """
    body = {
        "custom": False,          # inspiration mode
        "instrumental": False,    # vocals on
        "mv": "chirp-v5",         # default model from docs
        "gpt_description_prompt": (
            "Smooth late-night jazz with saxophone and piano, "
            "relaxed swing groove, emotional but calm atmosphere"
        ),
        # all other fields left at defaults
    }

    for attempt in range(max_retries):
        resp = requests.post(BASE_URL, headers=HEADERS, json=body, timeout=60)
        if resp.status_code in (429, 499):
            wait = 10 * (2 ** attempt)
            print(f"Rate limited ({resp.status_code}), retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        print("Submit response:", data)
        job_id = data["data"].get("job_id") or data["data"].get("jobId")
        if not job_id:
            raise RuntimeError(f"job_id not found in response: {data}")
        return job_id

    raise RuntimeError("Failed to submit job after max retries")


def fetch_job(job_id: str, max_retries: int = 60, delay: int = 5):
    """
    Poll TTAPI job result.
    """
    fetch_url = "https://api.ttapi.io/suno/v1/fetch"
    for _ in range(max_retries):
        resp = requests.get(
            fetch_url,
            headers=HEADERS,
            params={"jobId": job_id},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        print(f"Job {job_id} status:", status)

        if status == "SUCCESS":
            return data
        elif status in ("FAILED", "ERROR"):
            raise RuntimeError(f"Job {job_id} failed: {data}")
        time.sleep(delay)

    raise TimeoutError(f"Job {job_id} did not finish in time")


def download_tracks_from_result(data, job_index: int):
    musics = data.get("data", {}).get("musics", [])
    if not musics:
        print("No musics found in result")
        return

    for i, track in enumerate(musics):
        audio_url = track.get("audio_url") or track.get("audioUrl")
        title = track.get("title") or f"jazz_test_job{job_index}_track{i+1}"
        safe_title = "".join(
            c for c in title if c.isalnum() or c in (" ", "_", "-")
        ).rstrip()
        filename = OUTPUT_DIR / f"{safe_title}.mp3"
        if filename.exists():
            filename = OUTPUT_DIR / f"{safe_title}_job{job_index}_track{i+1}.mp3"
        print("Downloading:", audio_url, "->", filename)

        r = requests.get(audio_url, timeout=120)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)

JOBS_FILE = Path("job_ids.txt")


def main():
    # 50 jobs * 2 tracks per job = 100 songs
    num_jobs = 50

    # Resume from previously saved job IDs if the file exists
    if JOBS_FILE.exists():
        job_ids = JOBS_FILE.read_text().splitlines()
        print(f"Resuming with {len(job_ids)} existing job IDs from {JOBS_FILE}")
    else:
        job_ids = []

    print("Submitting jobs...")
    while len(job_ids) < num_jobs:
        job_id = submit_inspiration_job()
        job_ids.append(job_id)
        JOBS_FILE.write_text("\n".join(job_ids))  # save after every submission
        time.sleep(2)  # be nice to the API

    print("\nFetching and downloading results...")
    for idx, job_id in enumerate(job_ids, start=1):
        result = fetch_job(job_id)
        download_tracks_from_result(result, idx)

    JOBS_FILE.unlink(missing_ok=True)  # clean up after successful run

if __name__ == "__main__":
    main()