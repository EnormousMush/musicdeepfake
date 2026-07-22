# Flow-Matching / reconstruction-based detection — research notes

Status: initial survey + PoC design (2026-07). Exploratory branch (mentor's core idea);
NOT on the AASIST critical path. Companion to `roadmap.md` Phase 3e.

## The idea (mentor's framing)
"Human audio isn't on the AI generation path." Use a *generative model's own machinery*
to test whether a clip lies on the AI-generation manifold — via reconstruction or likelihood —
instead of a purely discriminative classifier on features.

## What the literature actually does (two key papers, read 2026-07)

### Afchar / Deezer, ICASSP 2025 — *AI-Generated Music Detection and its Challenges*
(arXiv 2501.10111 · github.com/deezer/deepfake-detector)
- **Method:** train a binary classifier on **(real track) vs (its own autoencoder reconstruction)**.
  Autoencoders: EnCodec (3/6/24 kbps), DAC/LAC, GriffinMel, Musika. 25k FMA tracks × (orig + 9
  reconstructions) = 250k. Best input = amplitude spectrogram; 6-conv CNN.
- **De-confound trick (the important part):** reconstructions stored at the **same bitrate** as the
  original → content + codec held constant → the classifier can *only* pick up **neural-decoder
  artifacts** (e.g. transposed-conv checkerboard). AI music (from a neural decoder) carries them.
- **Results:** 99.8% (real vs reconstruction); **99.9% on unseen MusicGen**.
- **Caveats (their own):** (1) fails under pitch-shift / white-noise / re-encoding; (2) **cross-decoder
  transfer collapses across families** (GriffinMel→DAC ≈ 0%) — it learns *decoder-specific* fingerprints,
  not general AI artifacts; (3) defaults to "real" when artifacts are masked; (4) needs constant updates.

### *Diffusion Reconstruction towards Generalizable Audio Deepfake Detection* (arXiv 2604.26465)
- **Method:** reconstruct audio through **SemantiCodec** (a latent-diffusion codec); use the
  reconstructions as **hard samples** for contrastive learning. Back-end = frozen **XLS-R 300M →
  AASIST** + a regularization-assisted contrastive loss (RACL). (Reconstruction is *augmentation*,
  not a direct error-score.)
- **Results (speech; ASVspoof/CodecFake):** avg EER over 5 sets 15.8% → 12.2% (diffusion) → **8.2%**
  (RACL+diffusion). Cross-generator (CodecFake) 36.8% → **20.2%** — improved, not solved.

## 🔑 Why this matters for OUR confound problem
Afchar's **"each track vs its own reconstruction"** is a *principled de-confounder*: same content,
same production, same codec — the ONLY difference is the generation artifact. Our Suno-vs-FMA task is
confounded precisely because content/production/era all differ. The reconstruction framing sidesteps
that by construction. So this branch is not just "another detector" — it's a candidate route *around*
the confound that's blocking our discriminative pipeline.

## PoC designs (by increasing risk)

**PoC-A — Afchar-style artifact classifier (most proven; we already have EnCodec):**
1. Human (FMA/MTG) + Suno clips.
2. Reconstruct each through **EnCodec-24kHz** (already in our `.hfcache`), store at matched settings.
3. Train a classifier on (original) vs (its reconstruction) — content/codec locked, only decoder
   artifact remains.
4. Apply to Suno: is it flagged as "carries neural-decoder artifacts"? Then **cross-generator: ACE-Step**.
- *Catch:* Suno's decoder ≠ EnCodec; per Afchar, cross-family transfer is weak → may not flag Suno unless
  the reconstruction family covers Suno's. Test explicitly.

**PoC-B — mentor's pure "on-manifold" score (more elegant, more exploratory):**
1. A flow-matching / diffusion audio model.
2. Per clip: partial-noise → denoise round-trip; score = reconstruction error (or likelihood).
3. AI (on-manifold) → low error; human (off-manifold) → high error.
- *Catch:* the error may itself be production-confounded (a model trained on clean modern audio
  reconstructs clean audio better). Needs a content-matched control to be trustworthy.

## Honest assessment
- Reconstruction/flow methods carry the **same cross-generator generalization risk** we already worry
  about (decoder-specific fingerprints). Not a silver bullet.
- Their real value for us is the **de-confounding construction** (track vs its own reconstruction),
  which we can borrow regardless of whether we adopt the full paradigm.
- Keep as an **exploratory branch**; PoC-A is cheap (EnCodec in hand) and worth a small run once the
  discriminative eval is de-confounded.

## Related (for the lit review)
- ArtifactNet — forensic residual physics for AI-music detection (arXiv 2604.16254)
- MusicDET — zero-shot AI-music detection (arXiv 2605.18072)
- *The AI Music Arms Race* — TISMIR survey (tismir.254)
- DFADD — diffusion/flow-matching TTS deepfake dataset (arXiv 2409.08731)
