# Testing / Experiment Plan

Status: **draft for review** · Last updated: 2026-07-11 · Companion to `design.md`.

How we test encoders, classifiers, and the adversarial loop **without** a combinatorial
explosion, and how we run experiments given that **Claude Code cannot reach the UChicago
server — only the user's terminal can** (I write scripts + a runbook; the user runs them on
the server and sends results back).

Data decisions in force: human = **FMA** (local hard drive), AI = **Suno** (local hard
drive), **instrumental-only**, first subset **a few thousand, balanced, genre-agnostic**.

---

## 1. Guiding rule: coordinate ascent, not grid search

Full grid = encoders (~4) × layers (~13 each) × classifiers (~3) × adversarial = explosion.
Instead **vary one axis at a time, freeze the rest**, using a cheap probe as the measuring
stick. Each stage's winner is frozen and carried into the next. Cost ≈ `#encoders + #classifiers
+ a few experiments`, not the product. This is the standard SSL layer-probing methodology.

## 2. Fixed foundations (set once; never change mid-comparison)

- **Data:** the frozen subset manifest (few-k FMA-instrumental + Suno-instrumental, balanced).
- **Split:** train/val/test frozen once; paired samples kept together; **test set untouched
  during development** (only for final numbers).
- **Metric:** EER (primary), AUC (secondary). One shared eval function.
- **Preprocess:** one canonical chain (mono → resample → crop → LUFS → codec-neutral),
  identical for every experiment.
- **Results log:** one CSV row per run — `exp_id, encoder, layers, classifier, params,
  eer_val, eer_test, notes`. This table is the project's memory of what worked.

## 3. Stages

| Stage | Vary | Freeze | Output | Where |
|-------|------|--------|--------|-------|
| **0 — plumbing + baseline** | mel + linear probe | — | pipe works, baseline EER, shortcut check | Mac CPU |
| **1 — encoder + layer** | encoders one at a time; layers swept cheaply from cache | classifier = linear probe | ranked encoders + best layer(s) each | server GPU |
| **2 — classifier** | linear → AASIST → SpecTTTra | encoder = stage-1 winner | best backend | server GPU |
| **3 — encoder fusion** | concat top-2 encoders | classifier = stage-2 winner | does fusion beat best single? | server GPU |
| **4 — robustness / generalization** | confound controls + cross-generator | champion config | is the signal real? | server GPU |
| **5 — adversarial** | one generator at a time (ACE-Step first) | frozen champion detector | co-evolution curve + humanness | server GPU |

**Stage 1 detail (the "one encoder at a time" answer).** Yes — encoders are evaluated one at
a time, but **layers are not re-extracted per experiment**. For each encoder, extract **all
layers once** and cache them; then run a cheap **linear probe per layer** → a layer-vs-EER
curve → pick the best layer(s). The classifier is deliberately the cheapest possible so the
comparison **isolates the encoder's contribution**. Then compare encoders at their best layer
under the same probe, and rank.

**Stage 2 detail.** Fix the winning encoder's features; swap classifiers (linear → AASIST →
SpecTTTra). Encoder constant ⇒ isolates the backend. AASIST/SpecTTTra need the temporal
dimension, so extract temporal features for the winning encoder here (pooled features sufficed
for stages 0–1).

**Stage 3 detail.** Concatenate the top-2 encoders (e.g. MERT + wav2vec2) and re-run the best
classifier. Keep fusion only if it beats the best single encoder — avoids redundancy (the
wavLM/wav2vec2 overlap concern).

**Stage 4 detail (most important — do before trusting any number).**
- **Confound controls:** re-run with loudness matched / codec matched / duration matched. If
  EER collapses toward ~50% after matching, the detector was exploiting a shortcut, not real
  AI artifacts. If it holds, the signal is real. (Directly targets the FMA-vs-Suno confound
  risk from `design.md §1`.)
- **Cross-generator:** hold out non-Suno AI (ACE-Step / YuE) as a **test-only** set and measure
  the EER drop. Tells us whether the detector learned "AI" or just "Suno's fingerprint."

**Stage 5 detail (adversarial — only after a champion detector exists).** You need a strong
frozen detector to attack. Then: score generations with it (judge) → mine hard samples (ones
that fool it) → LoRA fine-tune ACE-Step toward that direction → regenerate → re-score → iterate.
Track **EER vs. adversarial round**, and use the **62 hand-crafted features** to see *which*
features shift (the humanness answer). Stop when the detector catches up / the generator is
exhausted / quality collapses. Generators are added one at a time (ACE-Step first); this is a
co-evolution curve, not a grid.

## 4. The order, stated directly

Do **not** test every encoder × classifier pair. Sequence:
1. cheap probe ranks encoders + layers,
2. best encoder → choose the classifier,
3. optionally fuse the top encoders,
4. stress-test the winner (confounds + cross-generator),
5. attack the winner (adversarial).

## 5. Server workflow (I write, you run)

Claude Code cannot access `UCHI_HOST`; only the user's terminal can. So every
experiment ships as: a **self-contained script + a config file + pinned requirements + a
one-line command in `RUNBOOK.md`**. The user runs it on the server and sends back the
**results CSV / logs**; I read them and decide the next config. Constraints this imposes on
the code: no hardcoded local paths (paths via config/env), deterministic seeds, resumable,
and it must log the results-table row itself.

Stage 0 is the exception — it runs on the Mac (CPU) where the data and I both are.

## 6. Instrumental data-prep note

Both sides must be instrumental for a fair comparison. Suno is already instrumental
(`VOCAL_NEGATIVE_TAGS`). **FMA must be filtered to instrumental** — via FMA metadata where
available, otherwise a lightweight vocal-activity screen (drop tracks with strong vocal
energy). **Decision needed:** how strict (metadata-only, or an actual vocal detector, and the
threshold).
