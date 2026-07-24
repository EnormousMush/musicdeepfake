
# musicdeepfake
deepfake detector specialized in Suno-generated music

Deepfake detection for AI-generated music (primarily [Suno](https://suno.com)) vs. human music.
UChicago honors thesis · advisor Prof. Blase Ur.

The project is organized into four parts (Part 4 has no code yet).

## Repository layout

```
musicdeepfake/
├── part1_extraction/         # Part 1 — Suno-via-TTAPI extraction pipeline (resumable, reproducible)
│   ├── run.py                # entry point
│   ├── prompts.py            # prompt generation + freeze
│   ├── api.py                # TTAPI client
│   ├── submit.py poll.py downloader.py orchestrator.py
│   ├── manifest.py validate.py
│   ├── configs/              # per-batch JSON configs
│   ├── manifests/            # CSV manifests + frozen prompts
│   ├── word_lists/           # per-genre subgenre/mood/descriptor/tag lists
│   └── README.txt            # pipeline details
│
├── part2_analysis/           # Part 2 — preprocessing + 62 hand-crafted features (importable package)
│   ├── audio.py manifest.py  # canonical load / dataset manifest
│   ├── export_subset.py preprocess_suno_server.py
│   ├── analysis_utils.py     # shared: load/HPSS/plotting
│   ├── dynamics.py spectral.py timbral.py rhythm.py key.py
│   ├── quantize_deg.py lyrics_structure.py
│   └── genre_id/             # dataset genre-determination snippets (FMA / MTG / MagnaTagATune)
│
├── part3_detector/           # Part 3 — detector: frozen SSL encoders + probes + diagnostics
│   ├── run_stage0.py run_stage1.py check_server.py
│   ├── encoders/             # ssl.py (mert/wav2vec2/xlsr/muq/encodec), mel.py
│   ├── classifiers/          # linear.py, temporal.py
│   ├── eval/                 # eer.py
│   ├── diagnostics/          # confound battery: genre_matched, mel_confound, shuffle_check,
│   │                         #   bandwidth_ablation, codec_history, spectral_match
│   ├── configs/ docs/        # round configs; plan/roadmap/design/testing/notes
│   └── RUNBOOK.md FUDAN_RUN.md
│
├── HANDOFF.md                # read-first state of the project for a fresh session
└── archive/                  # earlier test scripts, pre-Part-1 experiments, test audio
```

## The four parts

- **Part 1 — Data (`part1_extraction/`, main body done):** Suno AI tracks via TTAPI across 8 genres;
  human data from CC-licensed FMA (+ MagnaTagATune on hand as a production control).
- **Part 2 — Preprocessing + features (`part2_analysis/`, done, role redefined):** canonical
  preprocessing (24 kHz / mono / 30 s / LUFS) and the 62 hand-crafted features — **not the detector**;
  they serve as a data-quality cleaner, a baseline, and a "measuring ruler" for the adversarial analysis.
- **Part 3 — Detector (`part3_detector/`, active):** frozen multi-SSL-encoder frontend (MERT /
  wav2vec2 / MuQ / EnCodec, hierarchical layers) + linear probes today, stronger classifiers
  (AASIST / SpecTTTra) gated behind de-confounding the eval. Includes the confound-control
  diagnostics battery. Flow Matching is tracked as a separate paradigm. Metric: EER.
- **Part 4 — Adversarial (framework set):** detector-as-judge scoring → LoRA fine-tune ACE-Step
  → mine hard samples → retrain. Studies which artifacts are removable vs. stubborn ("humanness").

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Audio outputs, model weights, and secrets are gitignored — see `.gitignore`.
The Suno pipeline reads its API key from the `TTAPI_KEY` environment variable.
