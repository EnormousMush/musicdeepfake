# HANDOFF — read this first in a new session

Purpose: this file lets a fresh Claude Code session (or a new teammate) resume with full context,
even though the repo was moved off the iCloud Desktop and memory may not auto-load.

## TL;DR of where we are
We built an AI-music (Suno) vs human deepfake detector pipeline end-to-end and got the first real
results. Latest finding: MERT features separate the two classes at ~1.3% EER, but a battery of
control experiments shows this is **confound-driven (production/era), not proven AI-detection yet**.
Next real experiment: an **era/production-matched human set**.

## Round 1 status (2026-07-19) — SUPERSEDES stale details below
**Goal this round:** run the *whole real pipeline* end-to-end on a small dataset (3000 AI + 3000
human), full methodology, optimize as we go — NOT real training/adversarial yet. Plan is now in
**`docs/plan.md`** (+ visual `docs/plan.html`); it reorganizes the roadmap around "small data,
full rigor."

Done:
- **Data + model cache live on the Seagate drive** (NOT the Mac — internal disk was nearly full)
  at `"/Volumes/Seagate /frank-suno-round1/"` (`subset_export_round1/` + `hfcache/`). The repo has
  symlinks `data_store/subset_export_round1` and `.hfcache` → the drive, so local runs work when
  it's mounted. RUNBOOK rsyncs to the server straight from the drive staging dir.
- **Data**: round-1 export built + verified — self-contained, **4.6 GB, 6000 clips** = 3000 Suno +
  3000 FMA instrumental; 24k/mono/30s/LUFS FLAC; splits 70/15/15, Suno `_1/_2` pairs grouped,
  **0 leak**. Config: `configs/round1.yaml`.
- **Encoders**: all 4 wired into `encoders/ssl.py` + verified on CPU — `mert`, `wav2vec2`, `muq`
  (Tencent MuQ, `pip install muq`), `encodec` (codec probe), plus `xlsr`. `run_stage1.py` +
  `check_server.py` `--encoder` choices updated.
- **Model cache**: **2.4 GB**, all 4 models pre-downloaded — rsync it to skip HF downloads
  (essential on Fudan/China; else `export HF_ENDPOINT=https://hf-mirror.com`).
- **Mac venv** recreated + working (export + local smokes all pass).

Data findings:
- **FMA cannot be era-matched** to Suno: catalog is 2008–2017, `track_date_recorded` only 16%
  filled. So round-1 human = plain FMA = **era-CONFOUNDED on purpose** (fine for a pipeline run).
  Real era control needs a MODERN human source.
- **MTG-Jamendo is NOT on the drive** — the drive's `magnatagatune` (MagnaTagATune, ~2009 label
  music, still zipped, 5.6 GB) was the name mix-up. MagnaTagATune can be a free *production-quality*
  control; MTG-Jamendo still needs downloading for the *era* control.

Compute (the binding blocker):
- **Fudan lab GPU server obtained**: `FUDAN_USER@FUDAN_HOST` ("节奏节拍"). Internal 10.x IP →
  reachable only from the Fudan intranet; user is external → **BLOCKED** pending Fudan VPN / jump
  host / Tailscale (mentor asked). This is the real-GPU path.
- UChicago `super` still GPU-down + home-disk-full.

**Immediate next step:** get network access to the Fudan server (or fix super's GPU), then follow
`RUNBOOK.md` (now server-agnostic + China HF-mirror) to run all 4 encoders' per-layer probes on the
round-1 set. Data, code, and model cache are all staged and ready to ship.

---

## What this project is
UChicago honors thesis (advisor Blase Ur; Fudan senior as mentor). Detect AI-generated music,
primarily Suno, vs human music. Repo `EnormousMush/musicdeepfake`. Four parts: (1) data, (2) 62
hand-crafted features, (3) detector = frozen multi-SSL-encoder frontend + trained classifier
backend, (4) adversarial co-evolution + humanness analysis. Metric = EER (lower = more separable).

## The plan
Full 5-phase roadmap: **`docs/roadmap.md`** (data → encoder+confound-tests → classifier →
adversarial → locked full run). Framework/why: `docs/design.md`. Testing methodology:
`docs/testing-plan.md`. Read those three; they are the source of truth for direction.

## Done so far (with numbers)
- **Pipeline built** (`common/ preprocess/ encoders/ classifiers/ eval/`, `run_stage0.py`,
  `run_stage1.py`), manifest-driven, cached, config-selected.
- **Stage 0 (mel + logistic regression)**: full 4000-clip subset (2000 Suno + 2000 FMA
  instrumental), **test EER 5.67%**.
- **Stage 1 (MERT-v1-95M, all layers, linear probe)**, run on the server CPU (~5 h): **best layer
  6 (mid), test EER 1.33%** (layers 5 & 9 = 0.50%; last layer 12 worst). SSL ≫ hand-crafted;
  mid-layers beat the last (literature-consistent).
- **Confound battery** (all cheap, from cached features):
  - loudness (normalized), hi-freq/codec bandwidth, brightness/linear-tilt → all **ruled out**.
  - genre-matched (`diagnostics/genre_matched.py`) → matched EER 2.38% vs 1.33%; within-genre ~0%
    (classical/electronic/jazz = 0.00%). **Genre is NOT the confound.** Only pop = 12.5% (the most
    modern/produced FMA genre) — hinting the real confound is **production/era/fidelity**.

## Current scientific conclusion
FMA (old, lo-fi) vs Suno (clean, modern) is separable at a very deep level, but we cannot yet claim
"AI-detection" vs "old-recording-detection." Near-0% within a single generator is NORMAL in
anti-spoofing and does not prove generality. Two experiments will disentangle it (both are in the
proposal): (1) **era/production-matched human set** (filter FMA by release year, or add
MTG-Jamendo); (2) **cross-generator eval** (test the Suno-trained detector on ACE-Step / YuE).

## Immediate next step
Phase 1c/2c: build an **era-matched human subset** (FMA metadata has release year) and re-run MERT
layer-6 EER. Decisive question: does ~1.3% survive era-matching? If EER rises a lot → the score was
production confound. Do NOT invest in Phase-3 classifiers (AASIST) until the eval is de-confounded.
In parallel: chase the GPU driver fix (admin) and sync findings with the mentor.

## Where everything lives
- **Repo**: was `~/Desktop/frank-suno-backup`, being moved to `~/Developer/frank-suno-backup`
  (off iCloud — Desktop sync was interfering with git). Local commits only; no GitHub push needed
  (and push needs VPN, which we're avoiding).
- **Data (Mac)**: external Seagate drive — path is `/Volumes/Seagate ` **with a trailing space**
  (always quote it). FMA at `.../fma/fma_large/` (+ `fma_metadata/raw_tracks.csv`, instrumental =
  `track_instrumental==1`, 6045 present); Suno at `.../suno_audio/<genre>/<uuid>_{1,2}.mp3` (22k;
  `_1/_2` = paired samples, use UUID as group_id). Preprocessed subset export at
  `/Volumes/Seagate /subset_export` (+ `_mini`).
- **Server** `UCHI_HOST` (user `UCHI_USER`): **root/home disk is 100% FULL — never write
  to `~`.** Use `/mindata` (17 TB, 7 TB free). Work dir = `/mindata/frank-suno/detector/` (isolated;
  the existing `/mindata/frank-suno/part1_extraction` etc. is the live Part-1 project — don't touch).
  GPU (GTX 1070) is **down** (nvidia driver not loaded, no passwordless sudo → needs admin) → CPU
  torch for now. Raw Suno mp3s are already on the server (`part1_extraction/audio/<genre>/`), so we
  only upload FMA and regenerate Suno there (`preprocess_suno_server.py`). Every server shell must
  redirect caches off the full home disk: `export HF_HOME=/mindata/frank-suno/detector/.hfcache`
  etc. See `RUNBOOK.md`.
- **Memory files** (may not auto-load after the move) are at
  `~/.claude/projects/-Users-USER-Desktop-frank-suno-git-backup/memory/` — read `MEMORY.md`
  there for the index if resuming.

## Critical gotchas
- Seagate path has a **trailing space**: `"/Volumes/Seagate /..."`.
- `rsync` runs on the **Mac**; `python`/server paths run in the **server ssh session** — don't mix.
- Server **home disk full** → keep everything on `/mindata`, redirect HF/pip/tmp caches.
- MERT on CPU ≈ 5 s/clip (5 h/run). GPU fix would be ~50× faster — worth chasing.
- Everything is **resumable/cached**: re-running probes/experiments on cached features is instant.

## How to resume
1. `cd ~/Developer/frank-suno-backup` and start Claude Code there.
2. Point it at this file + `docs/roadmap.md`.
3. The Mac `.venv` breaks after the move (hardcoded paths) — recreate if you need to run locally:
   `python3 -m venv .venv && source .venv/bin/activate && pip install numpy scipy scikit-learn
   librosa soundfile pyloudnorm pyyaml pandas`. (Not needed for server work.)
