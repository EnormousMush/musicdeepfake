"""
Poll module: iterate submitted rows up to `batch_size`, POST to /fetch,
record musicIds and audioUrls on SUCCESS, mark done or failed.
"""
from datetime import datetime, timezone

from manifest import Manifest
from api import TTAPIClient


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def poll_batch(manifest: Manifest, client: TTAPIClient, batch_size: int) -> int:
    submitted = manifest.by_status("submitted")[:batch_size]
    if not submitted:
        return 0

    completed = 0

    for row in submitted:
        try:
            data = client.fetch(row["ttapi_job_id"])
        except Exception as exc:
            print(f"  [{row['genre']}] fetch error for {row['job_uid'][:8]}: {exc}")
            continue

        status = data.get("status")

        if status == "SUCCESS":
            musics = data.get("data", {}).get("musics", [])
            m1 = musics[0] if len(musics) > 0 else {}
            m2 = musics[1] if len(musics) > 1 else {}
            manifest.update(
                row["job_uid"],
                status="done",
                music_id_1=m1.get("musicId", ""),
                music_id_2=m2.get("musicId", ""),
                audio_url_1=m1.get("audioUrl", ""),
                audio_url_2=m2.get("audioUrl", ""),
                completed_at=_now(),
                error="",
            )
            manifest.save()
            completed += 1

        elif status in ("FAILED", "ERROR"):
            manifest.update(
                row["job_uid"],
                status="failed",
                error=f"TTAPI reported: {status}",
                completed_at=_now(),
            )
            manifest.save()
            print(f"  [{row['genre']}] job failed: {row['job_uid'][:8]}")

        # Any other status (PROCESSING, PENDING, etc.) — leave as submitted, poll next cycle.

    return completed
