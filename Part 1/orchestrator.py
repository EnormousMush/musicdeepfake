"""
Orchestrator: runs submit -> poll -> download in a continuous cycle
until the manifest is fully done and all audio is downloaded.
Designed to run unattended inside tmux on a remote server.
"""
import sys
import time
import threading

from manifest import Manifest
from api import TTAPIClient
from submit import submit_batch
from poll import poll_batch
from downloader import download_batch


def run(manifest: Manifest, client: TTAPIClient, config: dict) -> None:
    output_dir_path = __import__("pathlib").Path(config["output_dir"])
    c = config["concurrency"]
    dl_lock = threading.Lock()

    print("Orchestrator running. Press Ctrl+C to stop (progress is saved to manifest).")

    try:
        while True:
            counts = manifest.counts()
            pending    = counts.get("pending", 0)
            submitted  = counts.get("submitted", 0)
            done       = counts.get("done", 0)
            failed     = counts.get("failed", 0)
            downloaded = manifest.downloaded_count()

            print(
                f"\n[cycle] pending={pending}  submitted={submitted}  "
                f"done={done}  downloaded={downloaded}  failed={failed}"
            )

            if pending == 0 and submitted == 0 and done == downloaded:
                print("All jobs submitted, fetched, and downloaded. Exiting.")
                break

            n_sub  = submit_batch(manifest, client, config, c["submit_batch"])
            n_poll = poll_batch(manifest, client, c["poll_batch"])
            n_dl   = download_batch(manifest, output_dir_path, c["download_workers"], dl_lock)

            if n_sub or n_poll or n_dl:
                print(f"         +submitted={n_sub}  +fetched={n_poll}  +downloaded={n_dl}")
            else:
                print(f"         Nothing to do, sleeping {c['cycle_sleep']}s...")
                time.sleep(c["cycle_sleep"])

    except KeyboardInterrupt:
        print("\nInterrupted. All progress is saved in manifest.csv.")
        sys.exit(0)
