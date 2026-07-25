# HANDOFF — read this first (current as of 2026-07-23)

A fresh Claude Code session (or teammate) should be able to resume with full context from this file.
Real server credentials are NOT here (public repo) — they live in `part3_detector/FUDAN_RUN.local.md`
(gitignored). Server host/user appear as `FUDAN_HOST` / `FUDAN_USER` placeholders below.

## Stance right now (important)
The project is in an **exploratory phase**: we have a rough roadmap, but we're deliberately going
**one experiment at a time to find the highest-academic-value angle** — the space is wide open. If a
step turns out to be especially valuable, we may pivot and make *that* the thesis focus rather than
marching the full 4-part plan. **The current strength (mentor-endorsed) is the rigor of our confound
handling.** (The user tracks their own thinking in Obsidian; this file is the shared source of truth.)

## What this project is
UChicago honors thesis (advisor **Blase Ur**; a **Fudan senior** as mentor). Detect AI-generated
music (primarily **Suno**) vs human music. GitHub repo `EnormousMush/musicdeepfake` (**public**).
Four parts: (1) data extraction, (2) preprocessing + 62 hand-crafted features, (3) detector =
frozen multi-SSL-encoder frontend + trained classifier backend, (4) adversarial co-evolution +
"humanness" analysis. Metric = **EER** (lower = more separable).

---

## TL;DR of current state
Round-1 (small-scale full pipeline) is **validated end-to-end on the Fudan GPU**. Per-layer linear
probes across 4 encoders give near-0% EER — but we **richly confirmed this is production/era
CONFOUND, not AI-detection**. The gate now: **de-confound the eval before investing in the classifier
or adversarial parts.** Next = ① a cheap bandwidth-matched ablation, ② the decisive **cross-generator
test with ACE-Step**.

## What's been done — with numbers

**Round-1 dataset** (`configs/round1.yaml`): 3000 Suno + 3000 FMA instrumental, 24 kHz / mono / 30 s
(offset 10 s) / LUFS FLAC; splits 70/15/15, Suno `_1/_2` pairs grouped by UUID, **0 split leak**.
Self-contained export `subset_export_round1/` (4.6 GB, on the Seagate drive + on the Fudan server).

**Run on the Fudan GPU** (RTX 3060; old pinned stack — see Infra). Per-layer linear-probe **best
test EER**:

| encoder | best test EER | shape / note |
|---|---|---|
| **MuQ** | **0.00%** | music SSL — 0.00% at *every* layer incl. layer 0 |
| **MERT** | **0.67%** | music SSL — ~0.1–0.9% across all layers |
| **wav2vec2** | **1.78%** | speech SSL — **early layers best, deeper worse** |
| **EnCodec** | **3.67%** | codec probe (1 layer) |
| ~~XLS-R~~ | skipped | its HF model wasn't shipped to the offline server |

**Confound analysis (our strongest result so far):**
- *Per-layer shape:* MuQ/MERT near-0% at **all** layers (incl. the raw layer-0) = **gross-confound
  signature** (separable by almost anything). wav2vec2's **early-good / deep-bad** profile says the
  signal is **low-level acoustic** (production/fidelity), not high-level structure.
- *Low-level diagnostic* (8 interpretable features, `test` EER): **loudness controlled** (rms EER
  45.78% ≈ chance → LUFS works); strongest single tell = **hf>10 kHz** (EER 30.44%, **Suno brighter /
  fuller-band**); centroid/rolloff/bandwidth moderate. **All 8 together → 24.44% EER**, vs SSL 0–2%.
  → The separation is **NOT** a cheap single-feature confound; there's a **real but modest
  brightness/HF/bandwidth production gap**, and the SSL models capture something **much richer** than
  crude production stats (whether "fine production texture" or a "generation fingerprint" is unresolved).

**Earlier (pre-Fudan, for reference):** mel+logreg baseline 5.67%; an earlier MERT CPU run gave
~1.33% (slightly different setup); **genre ruled out** as the confound (genre-matched ≈ same EER).

## Current scientific conclusion
Round-1's near-0% is **production/era-confound-driven, not proven AI-detection**. Pipeline mechanics
are validated, **but encoder/layer selection can't be finalized on confounded data** (all encoders
saturate to ~0%, so you can't tell which is genuinely better at detecting AI). Do **NOT** invest in
Part-3 classifiers (AASIST/SpecTTTra) or Part-4 adversarial until the eval is de-confounded — a
stronger head on a confounded eval just optimizes the confound, and attacking a confound-detector
teaches the generator to "sound old/lo-fi," not to "sound human."

## Confound-control battery (the plan, systematized)
- **Controlled:** loudness (LUFS → confirmed near-chance); genre (earlier: not the confound).
- **Found, real, to control — era / production** (FMA old lo-fi vs Suno modern clean; Suno brighter/
  fuller-band): ① **bandwidth/spectral matching** (low-pass both to a common cutoff, re-extract one
  encoder, re-probe — signal-level era control); ② **reconstruction de-confounder** (Afchar-style:
  each track vs its OWN reconstruction locks content/production/era/codec → only the generation
  artifact remains); ③ era-matched modern human corpus (optional, lower priority — ceiling is limited).
- **AI-side — cross-generator:** ACE-Step (does a Suno-trained detector transfer? AI-general vs
  Suno-specific).
- **Criterion:** only claim "AI-detection" when EER **survives every control**; else it's "detecting
  the dataset."

## Plan / options going forward (pick by value, not by rote)
- **A · de-confound (the priority):**
  - **A1 — bandwidth-matched ablation** (cheap, minutes, on existing server data): low-pass Suno+FMA
    to a common cutoff → re-extract 1 encoder → re-probe. Tests if the HF/brightness gap drives the EER.
  - **A2 — cross-generator (ACE-Step)** (decisive): download ACE-Step **v1.5 2B** (fits the 3060's
    ~5 GB free; XL-4B needs ≥12 GB) on the Mac, ship to the Fudan server (HF/GitHub blocked there),
    generate ~300–500 instrumental clips across the 8 genres, preprocess identically, eval the
    Suno-trained probe on ACE-Step-vs-human. (Check if a pre-generated ACE-Step set exists to skip gen.)
- **B · exploratory (parallel, mentor's core idea):** Flow-Matching / reconstruction PoC — see
  `part3_detector/docs/flow-matching-notes.md`. Key insight: reconstruction ("track vs its own
  reconstruction") is a **principled de-confounder**; **PoC-A** uses the EnCodec we already have.
- **C · gated behind de-confound:** strong classifiers (AASIST / SpecTTTra head-to-head) → freeze
  champion → adversarial co-evolution + 62-feature humanness analysis.

## Repo structure (after the part1/2/3 reorg)
```
part1_extraction/   Suno TTAPI extraction pipeline (run/prompts/api/submit/poll/downloader/... )
part2_analysis/     62 hand-crafted features (spectral/timbral/dynamics/rhythm/key/...) + PREPROCESSING:
                    audio.py (canonical load), manifest.py, export_subset.py, preprocess_suno_server.py
                    (part2_analysis is an importable package: __init__.py)
part3_detector/     encoders/ssl.py (mert/wav2vec2/xlsr/muq/encodec; encode_all_layers + encode_frames),
                    encoders/mel.py, classifiers/{linear,temporal}.py, eval/eer.py, diagnostics/,
                    run_stage0.py, run_stage1.py, check_server.py, configs/, docs/, RUNBOOK.md, FUDAN_RUN.md
HANDOFF.md, README.md, requirements.txt   (root)
```
- **Cross-part imports:** `run_stage0.py` + `diagnostics/mel_confound.py` import `part2_analysis`
  (preprocessing is Part 2; repo root on sys.path). **`run_stage1.py` is self-contained** — needs only
  `part3_detector/` (that's why the Fudan flow rsyncs just part3_detector).
- **Docs** (`part3_detector/docs/`): `plan.md` (round-1 execution plan, +`plan.html`), `roadmap.md`
  (5-phase), `design.md`, `testing-plan.md`, `flow-matching-notes.md`.

## Infra / how to run
- **Fudan GPU server** (real values in `part3_detector/FUDAN_RUN.local.md`): 2× RTX 3060 12 GB (shared —
  use **GPU 1**, `CUDA_VISIBLE_DEVICES=1`), Ubuntu 20.04, driver **470 → CUDA 11.4 max**, system
  Python **3.8**. **Work dir `/home/FUDAN_USER/mnt/bl/runbao-du`** — `mnt` is root-owned; **`bl/` is the
  only world-writable spot** (688 G free); the root disk `/` is 100% full. **HF + GitHub are BLOCKED**
  on the server → ship models/code from the Mac, pip via **Tsinghua mirror**, run **offline**
  (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`). **Pinned old stack:** `torch 1.12.1+cu113`
  (from download.pytorch.org), `transformers 4.35.2`, `muq`, numpy 1.24.4. ⚠️ **The venv must be made
  with system `/usr/bin/python3` (3.8) — NOT the auto-active conda `base` python, which is too new for
  torch 1.12.1+cu113** (that was the one real setup gotcha). tmux session name = `r1`. Full step-by-step:
  `FUDAN_RUN.local.md`.
- **On the server now:** `$WORK/data_store/subset_export_round1` (6000 clips), `$WORK/.hfcache`
  (4 models), and **cached per-clip features** at `.../subset_export_round1/features/<encoder>/` →
  **re-running any probe is ~instant** (no re-encode).
- **Seagate drive (Mac):** path `"/Volumes/Seagate /"` (**trailing space — always quote**). Raw FMA
  (`fma/fma_large/` + `fma_metadata/raw_tracks.csv`, instrumental `track_instrumental==1`, 6045 present),
  Suno (`suno_audio/<genre>/<uuid>_{1,2}.mp3`), MagnaTagATune (zipped, ~2009 label music — a possible
  *production* control). Staging `frank-suno-round1/` (`subset_export_round1` 4.6 G + `hfcache` 2.4 G);
  repo has symlinks `part3_detector/data_store/subset_export_round1` + `.hfcache` → the drive. **Drive
  is usually ejected;** mount it to re-export or feed the server.
- **UChicago `super`:** GTX 1070 GPU-down + home-disk-full — backup option only.
- **Mac venv** (`.venv`, torch 2.8 / transformers 4.57 / py3.9): works, for local code tests only
  (not encoder runs — the encoder models live on the Seagate/`.hfcache` and the Mac is CPU).

## GitHub state (mind the divergence)
- **`EnormousMush/musicdeepfake` main is PUBLIC** and now has part1/part2/part3 (the part-3 detector
  was merged in via a clean branch). The `reorg-part2` branch (moving preprocessing into Part 2) was
  pushed — **check if it's merged**; if not, open the PR.
- **The local repo's `main` is an UNRELATED history to GitHub main** (this repo was `git init`'d
  separately). Sync pattern: cherry-pick local-only commits onto a branch off origin/main, then
  fast-forward push to main; NEVER force-push. Last synced 2026-07-25 (diagnostics + HANDOFF +
  README). GitHub main still carries ~520 legacy binaries (old archive plots, a few root mp3/png)
  that local never tracked — harmless; clean up in a dedicated commit someday.
- Docs use **placeholders** (`FUDAN_HOST`/`FUDAN_USER`/`UCHI_*`) so no server creds are public;
  `*.local.md` is gitignored.

## Immediate next actions
1. **A1 bandwidth ablation** — write the low-pass re-extract + re-probe CLI, run on the server (minutes).
2. **A2 ACE-Step** — set up download→ship→generate for the cross-generator test (the decisive one).
3. (Optional) save the mentor progress report as `part3_detector/docs/progress-2026-07.md`.

## Gotchas (quick list)
- Seagate path has a **trailing space**.
- Fudan: **use system python 3.8, not conda base**; work under `mnt/bl/`; HF/GitHub blocked → ship+offline+Tsinghua; GPU1.
- Everything **cached/resumable** — re-running probes on cached features is instant.
- `rsync` runs on the **Mac**; `python`/paths run in the **server ssh session** — don't mix them up.
