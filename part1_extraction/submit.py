"""
Submit module: iterate pending rows up to `batch_size`, POST to /music,
store jobId, mark submitted. Exponential backoff on rate limits.
Never re-submits a row that is not in 'pending' status.
"""
import time
from datetime import datetime, timezone

from manifest import Manifest
from api import TTAPIClient


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _instrumental_map(config: dict) -> dict[str, bool]:
    return {
        genre: gcfg.get("instrumental", False)
        for genre, gcfg in config["genres"].items()
    }


def submit_batch(
    manifest: Manifest,
    client: TTAPIClient,
    config: dict,
    batch_size: int,
) -> int:
    pending = manifest.by_status("pending")[:batch_size]
    if not pending:
        return 0

    inst_map = _instrumental_map(config)
    submitted = 0

    for row in pending:
        instrumental = inst_map.get(row["genre"], False)
        success = False

        for attempt in range(6):
            try:
                ttapi_job_id = client.submit(
                    prompt=row["prompt"],
                    tags=row["tags"],
                    mv=row["mv"],
                    instrumental=instrumental,
                )
                manifest.update(
                    row["job_uid"],
                    status="submitted",
                    ttapi_job_id=ttapi_job_id,
                    submitted_at=_now(),
                    error="",
                )
                manifest.save()
                submitted += 1
                success = True
                break

            except Exception as exc:
                code = getattr(getattr(exc, "response", None), "status_code", None)
                if code in (429, 499):
                    wait = 10 * (2 ** attempt)
                    print(f"  [{row['genre']}] rate limited ({code}), retry in {wait}s")
                    time.sleep(wait)
                else:
                    manifest.update(
                        row["job_uid"],
                        status="failed",
                        error=str(exc),
                        completed_at=_now(),
                    )
                    manifest.save()
                    print(f"  [{row['genre']}] submit failed: {exc}")
                    break

        if success:
            time.sleep(0.5)  # gentle pacing between submissions

    return submitted
