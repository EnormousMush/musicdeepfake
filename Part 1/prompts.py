"""
Prompt generator: builds prompts for each genre by rotating through
subgenre / mood / descriptor word lists via modular indexing so combinations
spread evenly and reproducibly across the job set.

Word lists are loaded from JSON files in word_lists/<genre>.json.
The full prompt list is frozen to frozen_prompts.csv before any API calls
so the run is fully reproducible even if the script is restarted.
"""
import csv
import json
import math
from pathlib import Path

from manifest import Manifest

FROZEN_COLUMNS = ["job_uid", "genre", "subgenre", "prompt", "tags", "mv", "pair_group"]


def _load_word_list(genre: str, word_lists_dir: Path) -> dict:
    path = word_lists_dir / f"{genre}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Word list for '{genre}' not found at {path}.\n"
            f"Create {path} with keys: subgenres, moods, descriptors, tags."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize(s: str) -> str:
    return s.lower().replace("-", "").replace(" ", "")


def _build_prompt(genre: str, subgenre: str, mood: str, descriptor: str) -> str:
    already_named = _normalize(genre) in _normalize(subgenre)
    label = subgenre if already_named else f"{subgenre} {genre}"
    return f"{label}, {mood} atmosphere, featuring {descriptor}"


def generate_for_genre(
    genre: str,
    n_jobs: int,
    mv: str,
    word_lists_dir: Path,
) -> list[dict]:
    """
    Return `n_jobs` prompt dicts for the genre. Modular indexing:
      subgenre   = subgenres[i % S]
      mood       = moods[(i // S) % M]
      descriptor = descriptors[(i // (S*M)) % D]
    This exhausts subgenre combinations before repeating.
    """
    wl = _load_word_list(genre, word_lists_dir)
    subgenres = wl["subgenres"]
    moods = wl["moods"]
    descriptors = wl["descriptors"]
    tag_pool = wl.get("tags", [[]])

    S = len(subgenres)
    M = len(moods)

    rows = []
    for i in range(n_jobs):
        subgenre = subgenres[i % S]
        mood = moods[(i // S) % M]
        descriptor = descriptors[(i // (S * M)) % len(descriptors)]
        tag_set = tag_pool[i % len(tag_pool)] if tag_pool else []
        tags = ", ".join(tag_set) if isinstance(tag_set, list) else str(tag_set)

        rows.append({
            "genre": genre,
            "subgenre": subgenre,
            "prompt": _build_prompt(genre, subgenre, mood, descriptor),
            "tags": tags,
            "mv": mv,
            "pair_group": genre,
        })
    return rows


def freeze_and_populate(
    config: dict,
    manifest: Manifest,
    word_lists_dir: Path,
    frozen_path: Path,
) -> None:
    """
    One-time setup: generate all prompts, freeze to CSV, populate manifest.
    Safe to call on every startup — idempotent once frozen_path exists.
    """
    if frozen_path.exists():
        _repopulate_if_needed(manifest, frozen_path)
        return

    print("Freezing prompts (first run)...")
    all_rows: list[dict] = []
    for genre, gcfg in config["genres"].items():
        n_jobs = math.ceil(gcfg["target_tracks"] / 2)
        rows = generate_for_genre(genre, n_jobs, config["model"], word_lists_dir)
        all_rows.extend(rows)

    with open(frozen_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["genre", "subgenre", "prompt", "tags", "mv", "pair_group"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Frozen {len(all_rows)} prompts to {frozen_path}")

    for row in all_rows:
        manifest.add(
            genre=row["genre"],
            subgenre=row["subgenre"],
            prompt=row["prompt"],
            tags=row["tags"],
            mv=row["mv"],
            pair_group=row["pair_group"],
        )
    manifest.save()
    print(f"Manifest populated with {len(manifest.rows)} pending rows.")


def _repopulate_if_needed(manifest: Manifest, frozen_path: Path) -> None:
    """If manifest is shorter than frozen_prompts (e.g. crash mid-populate), add missing rows."""
    with open(frozen_path, newline="", encoding="utf-8") as f:
        frozen = list(csv.DictReader(f))

    if len(manifest.rows) >= len(frozen):
        return

    print(
        f"Manifest has {len(manifest.rows)} rows but frozen has {len(frozen)}. "
        f"Repopulating missing rows..."
    )
    for row in frozen[len(manifest.rows):]:
        manifest.add(
            genre=row["genre"],
            subgenre=row["subgenre"],
            prompt=row["prompt"],
            tags=row["tags"],
            mv=row["mv"],
            pair_group=row["pair_group"],
        )
    manifest.save()
    print(f"Manifest now has {len(manifest.rows)} rows.")
