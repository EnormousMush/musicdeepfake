Part 1 — Suno Track Extraction Pipeline
========================================

ENTRY POINTS
  run.py        Main entry point. Loads config, initializes manifest, freezes prompts,
                then drives the full submit/poll/download loop until all tracks are done.
                Flags: --config (default: configs/config.json), --freeze-only.

  validate.py   Standalone validation script. Checks status breakdown, per-genre track
                counts vs targets, duplicate TTAPI job IDs, missing audio files, and
                orphaned MP3s on disk. Run at any time to inspect pipeline state.

PIPELINE MODULES
  api.py        TTAPI client wrapper. submit() POSTs a generation job to /suno/v1/music;
                fetch() POSTs to /suno/v1/fetch to poll job status. API key is read
                from the TTAPI_KEY environment variable — never hardcoded.

  manifest.py   CSV manifest manager. Tracks every job (pending → submitted → done/failed)
                with columns for job IDs, audio URLs, local paths, and metadata.
                Writes atomically via .tmp + rename to prevent corruption.

  prompts.py    Prompt generation. Reads word lists (subgenres, moods, descriptors) and
                builds prompts via modular indexing so all combinations are evenly covered.
                freeze_and_populate() writes frozen_prompts.csv once and fills the manifest.

  submit.py     Queue-aware job submitter. Reads how many jobs are in-flight and only
                submits enough new jobs to stay under the max_queue cap. Stops
                automatically after 3 consecutive 403 (account block) responses.

  poll.py       Polls submitted jobs via the TTAPI fetch endpoint. On SUCCESS, records
                music IDs and audio URLs into the manifest. Marks truly failed jobs as
                "failed" so they can be retried.

  downloader.py Downloads completed audio files concurrently (ThreadPoolExecutor).
                Validates each MP3 by magic bytes; skips files already on disk.

  orchestrator.py  Ties submit/poll/download into a continuous cycle with a 10-second
                   sleep when nothing is happening. Exits with a bell and "ALL DONE"
                   message once every job is complete.

SUBDIRECTORIES
  configs/      JSON config files (one per batch run: config.json, jazz_config.json, etc.)
  manifests/    CSV manifests and frozen prompt files for each batch run
  word_lists/   Per-genre JSON files listing subgenres, moods, descriptors, and tags
  audio/        Downloaded MP3s — gitignored (large binary files)
