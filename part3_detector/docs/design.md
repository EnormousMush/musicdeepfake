# Pipeline & Experiment Framework — Design

Status: **draft for review** · Owner: Frank · Last updated: 2026-07-11

This document describes the end-to-end framework we will build to (a) run the whole
detection pipeline and (b) experiment with encoders / classifiers / adversarial
strategies **behind stable interfaces**, so that "running the pipeline" and "running
an experiment" are the same activity: the pipeline is the skeleton, experiments swap
parts in fixed slots.

If you only read one section, read **[§8 What I need from you](#8-what-i-need-from-you)**.

---

## 1. Goal

Run the whole pipeline end-to-end on a **first real subset** — a few thousand tracks
(FMA human + Suno AI, balanced, genre-agnostic) — with a cheap encoder (mel-spectrogram)
and produce an EER number. This runs on **CPU** (mel over a few thousand clips is ~20 min
on a Mac), so the first EER does **not** wait for GPU. A 10-file plumbing dry-run precedes
the subset run as cheap insurance. Then swap in real encoders/classifiers by config and
scale on GPU — never by rewriting.

Success for this first pass is **not a good accuracy** — it validates that data flows
through every stage and the interfaces hold, and gives a baseline number.

> **Caveat:** FMA (real human recordings) vs Suno (generated) may yield a deceptively low
> EER from recording/mastering/era confounds rather than real AI-artifact detection. That
> is expected here; preprocess normalization (§6) plus era-matched and cross-generator eval
> address it later.

---

## 2. Core principle

> A **manifest-driven, content-addressed-cached, config-selected** pipeline.

Six design points make "run pipeline while experimenting" work:

1. **Manifest = single source of truth.** One row per track; columns accrete as it
   flows: `audio_id | source | path | genre | label | dur/sr/lufs/… | split | feat_cache_key`.
   Same philosophy as the existing `data/suno_extraction/manifest.py` (resumable, atomic
   writes), generalized to every source.
2. **Content-addressed caching.** Every expensive artifact (preprocessed tensor, encoder
   features) is written to disk keyed by `hash(audio_id + params)`. Extract MERT once →
   train 10 classifiers on it for free. This is what makes experiments cheap.
3. **Registries + config.** Encoders and classifiers are registered by name and selected
   by a config file. An experiment is a YAML/JSON, not a code change (see §5).
4. **Split discipline baked in.** Splits are computed once at the manifest level and stored
   as a column; paired Suno samples share a `group_id` → same split; genre-stratified. Every
   downstream stage filters by the split column, so **leakage is structurally impossible.**
5. **EER harness = the only judge.** One eval function, one frozen test set ("协议锁定 /
   unified EER"). Every experiment reports comparably.
6. **Adversarial reuses everything.** Part 4 adds only a `generate(ACE-Step)+LoRA` stage
   that feeds new rows into the same manifest; the detector scores them with the same eval;
   hard samples get labeled and fed back. No new plumbing.

---

## 3. Pipeline stages

Each stage is a module with a stable input/output contract. The manifest is the bus.

| # | Stage | Reads | Writes | GPU? |
|---|-------|-------|--------|------|
| ① | **ingest** | data sources | raw audio + manifest rows | no |
| ② | **screen** | manifest | basic-screen metrics + 62-feature screen → cleaned manifest | no |
| ③ | **preprocess** | cleaned manifest | canonical tensors (mono, resampled, cropped, LUFS-normalized, codec-neutral) | no |
| ④ | **encode** | tensors | cached encoder features `[T*, D*]` (per encoder + layer set) | yes (scale) |
| ⑤ | **classify** | features + split | trained classifier + val metrics | yes (train) |
| ⑥ | **eval** | classifier + test split | EER, cross-generator generalization | light |
| ⑦ | **adversarial** | detector + ACE-Step | LoRA-tuned generator, hard samples → back to ① | yes |

Support: `manifests/` + `splits`, `config/`, `common/` (io, hashing, logging).

Contracts (informal):

```
ingest:      source_cfg           -> rows: {audio_id, source, path, genre, label, meta...}
screen:      manifest             -> manifest + {dur, sr, ch, lufs, fmt, bitdepth, feat62..., keep:bool}
preprocess:  manifest[keep]       -> tensor cache keyed by hash(audio_id + preprocess_params)
encode:      tensor + encoder_cfg -> feature cache keyed by hash(audio_id + encoder + layers + preprocess_params)
classify:    features[split=train/val] + clf_cfg -> model artifact + val curve
eval:        model + features[split=test]        -> {eer, auc, per_genre, per_source}
```

---

## 4. Repository layout (as it grows)

Today the repo has `data/` (Part 1) and `features/` (Part 2). New modules slot in at the
top level next to them — no restructuring needed:

```
musicdeepfake/
├── data/                 # Part 1 — ingest (suno_extraction ✅; fma.py, mtg_jamendo.py, acestep.py TBD)
├── features/             # Part 2 — 62 hand-crafted features (cleaner / baseline / measuring ruler)
├── preprocess/           # ③ decode→mono→resample→crop→LUFS→codec-neutral      (NEW, later)
├── encoders/             # ④ MERT / wav2vec2 / MuQ / EnCodec wrappers + layer select (NEW, later)
├── classifiers/          # ⑤ linear-probe / AASIST / SpecTTTra                  (NEW, later)
├── eval/                 # ⑥ EER + cross-generator harness                      (NEW, later)
├── adversarial/          # ⑦ scoring / LoRA / ACE-Step loop                     (NEW, later)
├── common/               # manifest, splits, hashing, config, logging           (NEW, later)
├── configs/              # experiment configs (one file per experiment)         (NEW, later)
├── data_store/           # gitignored: raw/ processed/ features/ (audio+tensors)
├── docs/                 # this file + diagrams + progress notes
└── archive/              # old tests + 6 test-audio fixtures
```

(We only create a directory when it gets its first real file — per your "no empty
scaffolding" preference.)

---

## 5. What an experiment looks like

```yaml
# configs/exp_mert_aasist.yaml
data:
  sources:   [suno, fma]
  n_per_class: 500
  genres:    all
preprocess:
  sr: 24000          # MERT wants 24 kHz
  crop_s: 60
  loudness_lufs: -23
encoder:
  name:   mert
  layers: [4, 6, 8]  # hierarchical — mid layers, not just the last
classifier:
  name: aasist
eval:
  metric: eer
  test_split: frozen_v1
```

Run → the harness resolves caches, trains, and logs `EER=…` to a results table.
Trying "various encoders/decoders" = copy the file, change two lines.

---

## 6. Why it works, and where it can break

The four known domain risks are absorbed **structurally**, not by discipline:

| Risk | How the framework kills it |
|------|----------------------------|
| **mp3/wav codec confound** (model learns "detect compression") | Preprocess (③) forces one canonical decode + resample + LUFS (and, if needed, transcodes both classes to the same codec) **before** encoding. Shared by all data → neutralized once, centrally. |
| **Paired-sample leakage** | Split at `group_id` level in the manifest (④ discipline). |
| **Speech encoder fails on music** | Encoder A/B is a config change → you find the winner by **measured EER**, not by guessing. |
| **Overfitting to the Suno fingerprint** | Cross-generator eval: hold out a non-Suno test set (ACE-Step/YuE) the classifier never trains on. The test set is just a manifest filter, so this is free. |

**Confirmation criterion:** v0 flows ~40 clips end-to-end and prints a sane EER. That
validates the framework. Accuracy comes later, with real encoders on GPU.

---

## 7. Compute

| Runs on CPU (now, Mac) | Needs GPU (later, UChicago server) |
|------------------------|------------------------------------|
| ingest, screen, preprocess | MERT/SSL feature extraction at scale |
| 62-feature reuse | Training AASIST / SpecTTTra to convergence |
| **first subset: mel + linear probe + EER (~few k clips)** | **All of Part 4** (LoRA + ACE-Step) |

The first subset uses **mel-spectrogram + pooled features + linear probe**, which runs
entirely on the Mac CPU — so the first real EER does not depend on GPU access. Real
encoders (MERT + wav2vec2) come next and move to the server GPU where the data already is.
Pooled features (mean/std over time → a few KB/clip) keep storage trivial; temporal
features (needed by AASIST) are cached later, on the server.

---

## 8. What I need from you

**Decided:** human = **FMA** (already downloaded on your hard drive); AI = **Suno**; first
subset = **a few thousand tracks total, balanced, genre-agnostic**; first-pass encoder =
**mel-spectrogram** (CPU, so the first EER does not wait for GPU). Pixabay is skipped for now.

### A. To run the first subset (next) — I mainly need paths + go-ahead
- **Path to the FMA audio** on your hard drive (a folder — I'll walk it and sample from it).
- **Path to the Suno AI audio** (hard drive; if it's only on the server, see §D).
- Target size: I'll take **~1.5–2k per class (~3–4k total)**, balanced, random across genres.
  Tell me if you want a different count.
- No manual clip picking and no labels file needed: I derive the label from the source folder
  (FMA → human, Suno → AI) and build the manifest myself.

### B. Decisions (don't block the subset; shape the real dataset later)
1. **Vocals vs instrumental** — irrelevant for the FMA subset; needed before the full dataset.
   Which for the main run? (early code was instrumental-only; newest workflow says "with vocals".)
2. **Encoder v1** — after mel proves the plumbing, confirm **MERT + wav2vec2/XLSR** as the first
   real encoders (add MuQ / EnCodec once the harness is proven; drop wavLM — overlaps wav2vec2).

### C. To scale on GPU (later)
- Confirmed **GPU access** on `UCHI_HOST`: model, VRAM, hours/quota, scratch
  storage. Blocks MERT-at-scale, AASIST/SpecTTTra training, and all of Part 4.

### D. Access / logistics
- If the Suno (or FMA) audio is only on the server, tell me how you want to work: build/run on
  the server, or copy a few-thousand-track sample to the Mac. Default: **mel subset on the Mac
  now, MERT on the server when GPU is confirmed.**
- Confirm the GitHub remote (`EnormousMush/musicdeepfake`) when you want this repo connected
  (I'll ask before any push).

---

## 9. Proposed next steps

1. You give me the **FMA path** and **Suno path** (§8-A) and pick a subset size.
2. I build the pipeline: `common/` (manifest + split + hash) → `preprocess/` (decode→mono→
   resample→crop→LUFS) → mel encoder → pooled features → linear probe → `eval/` EER. I run a
   10-file dry-run, then the few-thousand subset on CPU, and report the **first real EER**.
3. Confirm GPU (§8-C) → swap encoder to **MERT** by config, extract on the server, re-run EER.
4. From there: encoder sweep → strong classifier (**AASIST**) → cross-generator eval → Part 4.
