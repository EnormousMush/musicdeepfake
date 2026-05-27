"""
Manifest: one CSV row per job. Every state change writes atomically to disk
so a crash loses at most the current in-flight operation.

Status lifecycle:  pending -> submitted -> done -> (audio_path_1/2 filled)
                                        -> failed
"""
import csv
import uuid
from pathlib import Path

COLUMNS = [
    "job_uid",
    "genre", "subgenre", "prompt", "tags", "mv",
    "status",           # pending | submitted | done | failed
    "ttapi_job_id",
    "music_id_1", "music_id_2",
    "audio_url_1", "audio_url_2",   # CDN URLs, populated on SUCCESS fetch
    "audio_path_1", "audio_path_2", # local file paths, populated after download
    "pair_group",
    "submitted_at", "completed_at", "error",
]


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        self._by_uid: dict[str, int] = {}

    def load_or_create(self) -> None:
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.rows = [dict(row) for row in reader]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_uid = {r["job_uid"]: i for i, r in enumerate(self.rows)}

    def add(self, **fields) -> dict:
        row = {col: "" for col in COLUMNS}
        row["job_uid"] = str(uuid.uuid4())
        row["status"] = "pending"
        row.update(fields)
        self.rows.append(row)
        self._by_uid[row["job_uid"]] = len(self.rows) - 1
        return row

    def update(self, job_uid: str, **fields) -> None:
        idx = self._by_uid[job_uid]
        self.rows[idx].update(fields)

    def by_status(self, status: str) -> list[dict]:
        return [r for r in self.rows if r["status"] == status]

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        tmp.replace(self.path)  # atomic on POSIX

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    def downloaded_count(self) -> int:
        return sum(1 for r in self.rows if r.get("audio_path_1"))
