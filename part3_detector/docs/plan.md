# Execution Plan — Round 1: full real pipeline at small scale

Status: **living plan** · Companion to `roadmap.md` (vision), `design.md` (why), `testing-plan.md` (methodology).

## Goal of this round

Run the **entire real pipeline end-to-end** — raw audio → final EER (+ cross-generator + adversarial
experiment) — with the **full real methodology, nothing stubbed or minimized**, and **optimize the
pipeline as we go**. The *only* thing scaled down this round is the dataset: **AI 3000 + human 3000**.

Not a smoke test. Not a dry run with placeholder stages. We do all the real work:
- 4 encoders (all layers, per-layer probe, fusion) and real encoder/layer selection;
- the full confound battery, feeding back to improve Part 1 data;
- a real head-to-head classifier bake-off (AASIST vs SpecTTTra) to actually pick a champion;
- adversarial: this round is **experimental — the primary objective is to find + validate the right
  adversarial *method***, not to run it to convergence.

We run it like the real process, just on a small dataset the first time. Then we scale.

## Guiding rules (unchanged)

1. **One frozen EER harness is the only judge.** Fix it before touching models.
2. **De-confound before optimizing.** Two independent de-confound axes must both hold for the
   "AI vs human" claim: **human-side era/production matching** (FMA year filter + MTG-Jamendo) and
   **AI-side cross-generator** (Suno-trained detector tested on ACE-Step).
3. **Coordinate ascent** — fix each slot's winner, then move on. Confound tests (Part 2) feed back
   to Part 1 data; repeat until the EER is trustworthy.
4. **Everything config-driven and cached** — re-runs are cheap; one config reproduces the result.
5. **Compute is the binding constraint** — GPU driver down + server home disk full. Sequence heavy
   work behind GPU; chase the GPU fix in parallel. Scope stays full; only *timing* bends to compute.

Legend: ✅ done · 🔧 in progress · ⬜ not started · 🔒 blocked (GPU driver).

## Dataset spec (this round)

- **AI**: Suno, **3000** clips (already extracted on server). Paired samples (1 prompt → 2 songs)
  kept together in the same split (group by UUID) to prevent leakage.
- **Human**: **3000** clips from **FMA + MTG-Jamendo**, filtered by release year toward the Suno era
  (this *is* the era/production de-confound, built into the data from the start).
- **2nd AI source**: **ACE-Step**, pulled into Part 1 early (guards against a Suno-only fingerprint;
  feeds cross-generator eval and the adversarial experiment).
- **Instrumental only** this round (confirmed). Vocals deferred.
- **No genre stratification / matching** (confirmed): genre already ruled out as the confound; only
  total count matters.
- **Split**: 70/15/15, paired samples intact, no genre layer.

---

## Part 1 — Data: normalize + screen + era-matched, and prove it helps

- **1a. Preprocessing normalization + ablation.** mono / 24k / 60s (fixed offset, avoid intro &
  silence) / LUFS; **erase mp3↔wav codec differences** (else the model learns "detect mp3
  compression"). Prove via ablation that normalization removes shortcuts without killing real signal. 🔧
- **1b. 62 hand-crafted features as data cleaner.** Drop degenerate AI (silence/clipping/broken
  Suno); produce the **hand-crafted baseline EER**; freeze as the Part-4 measuring ruler. ⬜
- **1c. Era-matched human set.** FMA year filter toward Suno era + MTG-Jamendo. Makes the eval honest. 🔧
- **1d. Splits** — 70/15/15, paired-samples-together, no genre stratification. ⬜
- **1e. ACE-Step as 2nd AI source** pulled into Part 1. ⬜

**Done when:** a normalized, screened, era-matched 3000+3000 set + the 62-feature baseline number +
an ablation showing normalization/screening changed the confound picture.

## Part 2 — Encoders (all 4) + confound battery → feeds back to Part 1

Method = per-layer **linear probe + EER**.

- **2a. Four encoders, hierarchical (all layers).** MERT (24k) ✅ layer-6 1.33%; add **wav2vec2/XLSR
  (16k)**, **MuQ (24k, verify first — very new)**, **EnCodec (codec probe)**. wavLM cut (overlaps
  wav2vec2). Per-layer probe, pick best layer per encoder. **Frame-rate alignment to a common rate**
  before concatenation. 🔧🔒
- **2b. Fusion.** Concatenate top-N encoders/layers; keep only if it beats the best single. ⬜
- **2c. Full confound battery on the winner** (the important part): **era/production control**
  (old-FMA vs era-matched), **spectral-tilt control** at encoder level, **cross-generator**
  (Suno→ACE-Step). Each result **loops back to fix Part-1 data**. 🔧
- **2d. GPU** — unblock so 2a/2b aren't ~8h/encoder on CPU. 🔒

**Done when:** a chosen encoder set + layer(s) whose EER survives the controls (or a documented
residual confound + how Part 1 addresses it).

**Decision gate:** 2c controls must pass before investing in Part 3 — a stronger head on a
confounded eval just optimizes the confound.

## Part 3 — Classifier / detection head (NOT a "decoder"): real bake-off

Binary AI/human classifier. Terminology: it is a **classifier / detection head**, not a decoder.

- **3a. Linear probe baseline** ✅ (used throughout Part 2).
- **3b. W2V-AASIST baseline** (mentor's hard requirement): get an existing SOTA anti-spoof model
  actually running + EER logged. Provisional until 2c's controls pass. ⬜
- **3c. Temporal feature extraction** for the winning encoder(s) — frame-level, cached (AASIST /
  SpecTTTra need the time axis, not pooled mean/std). ⬜
- **3d. Strong classifier head-to-head (real selection):** **AASIST** (spectral-graph +
  temporal-graph attention) vs **SpecTTTra** (music transformer). Proper train/val/test, early
  stopping, same frozen EER judge, group split. Actually pick a champion. ⬜
- **3e. Re-run the confound battery on the strong head** — a stronger head exploits confounds
  harder; re-verify it isn't just a better shortcut-finder. ⬜
- **3f. Calibrate + freeze champion** — best de-confounded EER becomes the frozen detector Part 4
  attacks; record config. ⬜
- **3g. Flow-Matching track** (mentor's core idea, biggest open question): a *paradigm*, not a
  classifier — use the generative path / reconstruction as the detection signal ("human audio isn't
  on the AI generation path"); same family as Deezer/Afchar reconstruction-trace methods. Deep
  research + small proof-of-concept, parallel to the AASIST critical path. ⬜

**Done when:** a chosen classifier with the best de-confounded EER = the frozen champion.

## Part 4 — Adversarial: this round = find + validate the right METHOD (experimental)

Primary objective this round is **experimental**: identify and validate the right adversarial method,
not run it to convergence.

- **4a. Cross-generator eval first.** Champion tested on ACE-Step / YuE = the real "is it AI or just
  Suno?" test. ⬜
- **4b. Method exploration (main deliverable this round).** Candidate main line = **tier-2 score
  feedback**: detector scores generations → mine samples that fool it → **LoRA fine-tune ACE-Step**
  → regenerate → re-score. Validate the *loop mechanics* work end-to-end on small data. (Tier-1
  gradient-through-generator = future work; ACE-Step's discrete tokens + multi-step diffusion aren't
  fully differentiable.) ⬜
- **4c. Humanness signal via 62 features** — measure *which* features shift under adversarial
  pressure: removable = surface artifact, stubborn = deep fingerprint. ⬜
- **4d. Post-adversarial de-confound** — confirm we lowered real AI-ness, not learned a new data
  confound. ⬜

**Done when:** a validated adversarial method + first cross-generator numbers + a first read on which
artifacts are stubborn. (Real iteration to convergence = next round, at scale.)

## Closing — lock config + reproducible small run

Once each slot's winner is chosen (data recipe / encoder+layers / classifier / adversarial method),
freeze the config and confirm **one config reproduces raw audio → final EER + cross-gen +
humanness plots** on the 3000+3000 set. Then scale to full ~30–40k.

---

## Execution order (critical path; handles the GPU blocker)

**Now (CPU / cached, no GPU needed)**
1. Lock the **frozen EER harness + data recipe** (1a/1b/1d) — the judge for everything.
2. Build the **era-matched 3000 human set + 3000 Suno** (1c); ACE-Step into the pool (1e).
3. **Flow-Matching deep research** (3g) — pure understanding, no compute.

**GPU up (heavy)**
4. Four encoders × all layers → per-layer probe → fusion (2a/2b) → **full confound battery (2c)**,
   looping back to Part 1.
5. W2V-AASIST baseline (3b) → temporal features (3c) → AASIST vs SpecTTTra bake-off (3d/3e) →
   freeze champion (3f).

**Champion frozen**
6. Cross-generator eval (4a) → adversarial-method experiment (4b–4d).
7. Lock config, reproducible small run (Closing).

**Parallel throughout:** chase GPU (admin fixes nvidia driver) + keep everything on `/mindata`
(home disk full, redirect HF/pip/tmp caches) + sync findings with the mentor.

## Open items to confirm as we go

- **MuQ** — very new; verify details before wiring (2a).
- **Self-built vs SONICS (97k incl. Suno) vs combined** — this round uses self-built 3000+3000;
  SONICS is an option when scaling.
- **Compute** — server GPU model / hours / storage still unconfirmed; GPU driver down is the hard
  blocker on Parts 2–4.
- **Vocals vs instrumental** — instrumental this round; revisit before scaling.
