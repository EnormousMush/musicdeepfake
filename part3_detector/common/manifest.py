"""
Build the experiment manifest: sample a balanced instrumental subset of
Suno (AI) + FMA (human), assign label / group_id / split.

Manifest columns: audio_id, source, label, path, group_id, genre, split
  - label:    1 = AI (Suno),  0 = human (FMA)
  - group_id: Suno UUID (a _1/_2 pair shares one) or "fma_<track_id>".
              Split is assigned per group so paired samples never leak.
"""
import csv
import os
import random
from pathlib import Path

SUNO_GENRES = ["blues", "classical", "country", "electronic",
               "hiphop", "jazz", "pop", "rock"]


def _fma_path(audio_root: str, track_id: int) -> str:
    s = f"{int(track_id):06d}"
    return os.path.join(audio_root, s[:3], s + ".mp3")


def _collect_suno(suno_root: str):
    """Return list of (audio_id, path, group_id, genre) for all Suno tracks."""
    rows = []
    for genre in SUNO_GENRES:
        gdir = Path(suno_root) / genre
        if not gdir.is_dir():
            continue
        for f in gdir.glob("*.mp3"):
            stem = f.stem                      # "<uuid>_1"
            uuid = stem.rsplit("_", 1)[0]      # drop the _1/_2 suffix
            rows.append((f"suno_{stem}", str(f), uuid, genre))
    return rows


def _collect_fma(fma_meta: str, fma_audio_root: str):
    """Return list of (audio_id, path, group_id, genre) for instrumental FMA tracks present on disk."""
    rows = []
    with open(fma_meta, encoding="utf-8") as fh:
        r = csv.reader(fh)
        h = next(r)
        ti, ii = h.index("track_id"), h.index("track_instrumental")
        gi = h.index("track_genres") if "track_genres" in h else None  # genre unused in Stage 0
        for row in r:
            if row[ii].strip() != "1":
                continue
            tid = row[ti].strip()
            p = _fma_path(fma_audio_root, tid)
            if os.path.exists(p):
                genre = (row[gi].strip() if gi is not None and gi < len(row) else "") or "unknown"
                rows.append((f"fma_{tid}", p, f"fma_{tid}", genre))
    return rows


def _assign_splits(items, split_cfg, rng):
    """items: list of dicts sharing a class. Split by group_id so pairs stay together."""
    groups = {}
    for it in items:
        groups.setdefault(it["group_id"], []).append(it)
    gids = list(groups)
    rng.shuffle(gids)
    n = len(gids)
    n_tr = int(n * split_cfg["train"])
    n_va = int(n * split_cfg["val"])
    for i, gid in enumerate(gids):
        split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        for it in groups[gid]:
            it["split"] = split


def build_manifest(cfg) -> list:
    d = cfg["data"]
    rng = random.Random(cfg["seed"])
    n = d["n_per_class"]

    # --- AI: sample whole Suno pairs so a pair is never split ---
    suno = _collect_suno(d["suno_root"])
    by_group = {}
    for aid, path, gid, genre in suno:
        by_group.setdefault(gid, []).append((aid, path, gid, genre))
    gids = list(by_group)
    rng.shuffle(gids)
    ai_items = []
    for gid in gids:
        for aid, path, g, genre in by_group[gid]:
            ai_items.append(dict(audio_id=aid, source="suno", label=1,
                                 path=path, group_id=g, genre=genre))
        if len(ai_items) >= n:
            break
    ai_items = ai_items[:n]

    # --- human: sample FMA instrumental tracks ---
    fma = _collect_fma(d["fma_meta"], d["fma_audio_root"])
    rng.shuffle(fma)
    human_items = [dict(audio_id=aid, source="fma", label=0,
                        path=path, group_id=gid, genre=genre)
                   for aid, path, gid, genre in fma[:n]]

    _assign_splits(ai_items, d["splits"], rng)
    _assign_splits(human_items, d["splits"], rng)

    manifest = ai_items + human_items
    rng.shuffle(manifest)
    return manifest


def save_manifest(manifest, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = ["audio_id", "source", "label", "path", "group_id", "genre", "split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest)


def load_manifest(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
    return rows
