"""
Download module: for done rows with audioUrls and no local paths yet,
download both tracks, verify each is a non-zero valid MP3, record paths.
Idempotent — skips files already present and valid.
Uses a thread pool for parallelism; writes to manifest under a lock.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from manifest import Manifest

# MP3 magic bytes: ID3 tag or raw MPEG sync word
_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")


def _is_valid_mp3(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "rb") as f:
        header = f.read(3)
    return any(header[:len(m)] == m for m in _MP3_MAGIC)


def _download_one(url: str, dest: Path) -> bool:
    """Download url -> dest. Returns True if dest passes MP3 validation."""
    if _is_valid_mp3(dest):
        return True  # already downloaded and valid

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return _is_valid_mp3(dest)


def _process_row(row: dict, output_dir: Path) -> tuple[str, dict]:
    uid = row["job_uid"]
    genre_dir = output_dir / row["genre"]
    updates: dict = {}

    for slot in ("1", "2"):
        url = row.get(f"audio_url_{slot}", "")
        existing_path = row.get(f"audio_path_{slot}", "")
        if not url:
            continue
        if existing_path and _is_valid_mp3(Path(existing_path)):
            continue  # already done

        dest = genre_dir / f"{uid}_{slot}.mp3"
        try:
            ok = _download_one(url, dest)
            if ok:
                updates[f"audio_path_{slot}"] = str(dest)
            else:
                print(f"  [{row['genre']}] {uid[:8]}_{slot}: invalid MP3 after download")
        except Exception as exc:
            print(f"  [{row['genre']}] {uid[:8]}_{slot}: download error: {exc}")

    return uid, updates


def download_batch(
    manifest: Manifest,
    output_dir: Path,
    workers: int,
    lock: threading.Lock,
) -> int:
    to_download = [
        r for r in manifest.by_status("done")
        if r.get("audio_url_1") and not r.get("audio_path_1")
    ]
    if not to_download:
        return 0

    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_row, row, output_dir): row for row in to_download}
        for future in as_completed(futures):
            uid, updates = future.result()
            if updates:
                with lock:
                    manifest.update(uid, **updates)
                    manifest.save()
                downloaded += 1

    return downloaded
