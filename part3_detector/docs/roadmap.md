# Project Roadmap

Status: **living plan** · Last updated: 2026-07-11 · Companion to `design.md` + `testing-plan.md`.

The end goal: an AI-music (Suno) vs human detector whose reported number actually means
"detects AI," not "detects a dataset artifact" — plus an adversarial study of what "humanness"
is. Five phases, run as **coordinate ascent** (fix each phase's winner before moving on), with a
**feedback loop between Phase 1 (data) and Phase 2 (encoder + confound tests)**.

Guiding rules (unchanged): one frozen EER harness is the only judge; de-confound the eval
BEFORE optimizing the model; everything config-driven and cached so re-runs are cheap.

Legend: ✅ done · 🔧 in progress · ⬜ not started · 🔒 blocked (GPU driver).

---

## Phase 0 — Pipeline skeleton ✅
Built the manifest-driven, cached, config-selected pipeline: `common/ preprocess/ encoders/
classifiers/ eval/`, plus `run_stage0.py` (mel baseline) and `run_stage1.py` (SSL, server).
**Done.** mel baseline EER 5.67%; MERT EER 1.33%.

## Phase 1 — Training data: normalize + screen + prove it helps
**Goal:** a clean, confound-reduced dataset, plus a preprocessing + 62-feature screening step
we've *shown* to be useful (not assumed).

Steps:
- 1a. **Preprocessing normalization** (`preprocess/audio.py`) — mono/24k/30s/LUFS ✅. Add an
  optional **spectral-tilt / codec normalization** and prove via ablation it removes shortcut
  confounds without killing real signal. 🔧
- 1b. **62-feature screening as a data cleaner** (`features/`) — run the 62 hand-crafted
  features over the data and use them to (i) **drop low-quality / degenerate AI** (broken Suno
  outputs, silence, clipping), (ii) provide a **hand-crafted baseline EER**, (iii) be the
  **"measuring ruler"** for Phase 4. Prove useful = screening measurably improves data quality
  and the baseline is logged. ⬜
- 1c. **Better human data (from the Phase-2 confound finding)** — FMA is old/lo-fi; build an
  **era/production-matched** human set: filter FMA by release year (metadata has it) toward the
  Suno era, and/or add **MTG-Jamendo** (more modern). This is what makes the eval honest. ⬜
- 1d. **Vocals-vs-instrumental** decision for the full dataset (currently instrumental-only). ⬜

**Done when:** a documented, screened, era-matched training set + the 62-feature baseline number,
and an ablation showing normalization/screening changed the confound picture.

## Phase 2 — Best encoder(s) + confound testing → feeds back to Phase 1
**Goal:** pick the encoder(s)/layers whose EER *survives controls*. Method = linear probe + EER
(as you said). Confound tests here **feed back** to Phase 1 data.

Steps:
- 2a. **Encoder sweep** (all layers, per-layer probe, pick best layer): MERT ✅ (best = layer 6,
  1.33%). Run **wav2vec2/XLSR**, **MuQ** (verify first — very new), **EnCodec** (codec probe). 🔒(CPU-slow)
- 2b. **Encoder fusion** — concat top-2 (e.g. MERT+wav2vec2), keep only if it beats the best single. ⬜
- 2c. **Confound battery on the winner** (the important part): genre-matched ✅ (genre is NOT the
  confound); still to do — **era/production control** (compare EER on old-FMA vs era-matched human),
  **spectral-tilt control** at MERT level, and **cross-generator** (Phase 4). Each result loops
  back to improve Phase-1 data. 🔧
- 2d. **GPU** — unblock (admin fixes nvidia driver) so 2a/2b aren't 5 h/run. 🔒

**Done when:** a chosen encoder + layer(s) with an EER we trust because it holds under the controls
(or a documented account of the residual confound and how Phase 1 addresses it).

## Phase 3 — Classifier (the "detection head"; your "decoder")
**Goal:** the strongest backend on the chosen encoder features. (This is a *classifier* — binary
AI/human — not an audio decoder.)

Steps:
- 3a. **Linear probe** baseline ✅ (used throughout Phase 2).
- 3b. **Extract temporal features** for the winning encoder (AASIST/SpecTTTra need the time axis,
  not just pooled mean/std). Cache them like everything else. ⬜
- 3c. **Train strong classifiers** on the same features, compared head-to-head by EER:
  **AASIST** (SOTA anti-spoofing: spectral-graph + temporal-graph attention) and **SpecTTTra**
  (music-specific transformer). Proper train/val/test, early stopping. ⬜
- 3d. **Re-run the confound battery** on the strong classifier — a stronger head can exploit
  confounds harder, so re-verify it isn't just a better shortcut-finder. ⬜
- 3e. **Flow-Matching track** (mentor's core idea, exploratory / parallel) — use the generative
  path / reconstruction as a detection signal ("human audio isn't on the AI generation path").
  Deep research + a small proof-of-concept; not on the AASIST critical path. ⬜

**Done when:** a chosen classifier with the best de-confounded EER — this becomes the **frozen
"champion" detector** that Phase 4 attacks.

## Phase 4 — Adversarial co-evolution + humanness
**Goal:** answer "can the generator evolve to look more human, and which artifacts are removable
(surface) vs stubborn (deep machine fingerprint)?" — the thesis payoff.

Steps:
- 4a. **Add ACE-Step** (open-source, diffusion) as a generator + a **2nd AI data source** (pull it
  into Phase 1 too, to avoid a Suno-only fingerprint). ⬜
- 4b. **Cross-generator eval FIRST** (before any adversarial training): test the Suno-trained
  champion on ACE-Step / YuE. This is the real "is it AI or just Suno?" test and the entry to Part 4. ⬜
- 4c. **Adversarial loop, tier-2 (score feedback)** — detector scores generations → mine hard
  samples that fool it → **LoRA fine-tune ACE-Step** toward fooling → regenerate → re-score →
  iterate. (Tier-1 gradient-through-generator is future work — ACE-Step isn't fully differentiable.) ⬜
- 4d. **Humanness analysis** — at each round, use the **62 hand-crafted features** to measure
  *which* features shift. Removable = surface artifact; stubborn = deep fingerprint. ⬜
- 4e. **Stop criteria** — detector catches up / generator exhausted / quality collapses. ⬜

**Done when:** an empirical map of which AI artifacts survive adversarial pressure = the proposal's
answer to "what is humanness."

## Phase 5 — Lock config + full reproducible run
**Goal:** once every slot's winner is chosen (data recipe, encoder+layers, classifier, adversarial
protocol), freeze the config and run the **whole pipeline end-to-end on a small dataset** as a clean
integration test, then scale to the full ~30–40k.

**Done when:** one config reproduces the full result from raw audio to final EER + humanness plots,
runnable by someone else.

---

## Dependencies & cross-cutting
- **GPU driver** (admin) blocks the *speed* of Phases 2–4 (CPU works but ~5 h/run). Chase it.
- **Phase 1 ⇄ Phase 2 loop**: confound tests (2c) tell us how to fix the data (1c); repeat until
  the EER is trustworthy. Do NOT invest in Phase 3 classifiers until 2c's controls pass — a better
  classifier on a confounded eval just optimizes the confound.
- **62 features** thread through the whole project: Phase 1 cleaner, Phase 2/3 baseline, Phase 4
  humanness ruler.

## What's next (immediate)
1. Phase 1c / 2c — **era-matched human set** (FMA release-year filter) and re-run MERT layer-6 EER.
   The single most decisive next experiment: does the ~1.3% survive era-matching?
2. In parallel: chase the **GPU fix**, and sync findings with the mentor.
3. Then Phase 2a (wav2vec2, MuQ, EnCodec) once GPU is up.
