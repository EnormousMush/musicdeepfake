"""
Entry point.

Usage:
  python run.py                    # full run with config.json
  python run.py --config path.json # custom config
  python run.py --freeze-only      # freeze prompts + populate manifest, then exit
"""
import argparse
import json
import os
import sys
from pathlib import Path

from manifest import Manifest
from prompts import freeze_and_populate
from api import TTAPIClient
from orchestrator import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Suno track extraction pipeline")
    parser.add_argument("--config", default="configs/config.json", help="Path to config JSON")
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="Freeze prompts and populate manifest, then exit without making API calls",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    api_key = os.environ.get(config["api_key_env"])
    if not api_key and not args.freeze_only:
        sys.exit(
            f"Environment variable '{config['api_key_env']}' is not set. "
            f"Export it before running:\n  export {config['api_key_env']}=your_key"
        )

    manifest = Manifest(Path(config["manifest_path"]))
    manifest.load_or_create()

    freeze_and_populate(
        config=config,
        manifest=manifest,
        word_lists_dir=Path(config["word_lists_dir"]),
        frozen_path=Path(config["frozen_prompts_path"]),
    )

    if args.freeze_only:
        counts = manifest.counts()
        print(f"Done. Manifest: {counts}. Exiting (--freeze-only).")
        return

    client = TTAPIClient(api_key)
    run(manifest, client, config)


if __name__ == "__main__":
    main()
