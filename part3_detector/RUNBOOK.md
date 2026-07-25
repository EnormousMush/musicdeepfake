# Server Runbook — Stage 1 (SSL / codec encoders), Round 1

Claude Code **can't reach either GPU server** (both are behind institutional networks), so
**you** run these from your terminal and paste `results_stage1.csv` + the per-layer table back
to Claude.

This runbook is **server-agnostic**. Two targets so far — pick yours and read its box:

| Target | Host | Access | Notes |
|---|---|---|---|
| **UChicago** `super` | `UCHI_USER@UCHI_HOST` | campus network / VPN | root disk 100% full → everything on `/mindata`; GPU driver down → CPU torch |
| **Fudan** `FUDAN_USER` | `FUDAN_USER@FUDAN_HOST` | **Fudan intranet only** (internal 10.x IP) → need Fudan VPN / jump host / Tailscale | China network → **use HF mirror or ship the model cache** (see §3) |

Artifacts this runbook moves. **The heavy ones live on the Seagate drive** (mount it first — the
volume name has a trailing space: `"/Volumes/Seagate /"`) under a staging dir
`"/Volumes/Seagate /frank-suno-round1/"`. The code stays in the Mac git repo
(`~/Developer/frank-suno-backup/part3_detector`); the repo has symlinks `.hfcache` and
`data_store/subset_export_round1` pointing at the drive, so local runs work when it's mounted.
- **Data**: `frank-suno-round1/subset_export_round1/` — self-contained, **4.6 GB**, 6000 clips
  (3000 Suno + 3000 FMA instrumental), `audio/*.flac` + `manifest.csv`. 24 kHz mono, 30 s, LUFS.
- **Model cache**: `frank-suno-round1/hfcache/` — **2.4 GB**, all 4 encoder models pre-downloaded
  (MERT, wav2vec2-base, MuQ, EnCodec-24khz). Ship this to skip HF downloads entirely.
- **Encoders wired**: `mert`, `wav2vec2`, `xlsr`, `muq`, `encodec` (all verified on CPU).

---

## 0. Per-shell environment (redirect caches off the root disk)

Set a `WORK` dir on a big disk, and point all caches at it. **Run in every server shell**
(or append to `~/.bashrc`).

```bash
# --- super (UChicago): root disk is FULL, use /mindata ---
export WORK=/mindata/frank-suno/detector
# --- fudan: pick a scratch/data dir with space (confirm with `df -h`), e.g. ---
# export WORK=$HOME/frank-suno/detector

export HF_HOME="$WORK/.hfcache"
export TMPDIR="$WORK/.tmp"
export PIP_CACHE_DIR="$WORK/.pipcache"
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
```

⚠️ **super only**: the root disk (incl. `/home/UCHI_USER`) is 100% full — NEVER write there
(no venv, data, or caches). Keep everything under `/mindata` (17 TB). Don't touch the existing
`part1_extraction/`, `part2_analysis/`.

---

## 1. Copy code to the server (from the Mac)

```bash
cd ~/Developer/frank-suno-backup/part3_detector
rsync -av --exclude .venv --exclude .git --exclude data_store --exclude .hfcache \
  ./ <user@host>:"$WORK"/
```
✅ Check: `ssh <user@host> 'ls "$WORK"'` shows `run_stage1.py`, `encoders/`, `requirements-server.txt`.

---

## 2. Set up the env

```bash
cd "$WORK"
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# torch: match the box. GPU (see nvidia-smi CUDA), e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only fallback (super, GPU down):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-server.txt
```
`run_stage1.py` auto-uses CUDA if available, else CPU.

---

## 3. Get the encoder models — two ways

**(a) Ship the pre-downloaded cache — RECOMMENDED for Fudan / China** (no HF access needed):
```bash
# from the Mac (source is the Seagate staging dir; note the trailing space in the volume name)
rsync -av "/Volumes/Seagate /frank-suno-round1/hfcache/" <user@host>:"$WORK"/.hfcache/
```

**(b) Let the server download them** — fine on `super`; on Fudan set the mirror first:
```bash
export HF_ENDPOINT=https://hf-mirror.com     # China: HuggingFace is blocked/slow without this
# models download on first use into $HF_HOME
```

---

## 4. Get the data onto the server (from the Mac)

The round-1 export is **self-contained** (both FMA and Suno FLACs already inside), so just
copy the whole folder — no server-side regeneration needed.
```bash
# source is the Seagate staging dir (trailing space in the volume name is real)
rsync -av "/Volumes/Seagate /frank-suno-round1/subset_export_round1" \
  <user@host>:"$WORK"/data_store/
```
✅ Check: `ls "$WORK"/data_store/subset_export_round1/audio | wc -l` → **6000**;
`wc -l "$WORK"/data_store/subset_export_round1/manifest.csv` → **6001** (6000 + header).

---

## 5. Verify (zero / tiny data)

```bash
python check_server.py --encoder mert           # loads model, prints feature shape
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder mert --limit 50
```
✅ `--limit 50` prints a per-layer EER table (50-clip numbers are meaningless — only that it runs).

---

## 6. Run — quick pass, then full (use tmux)

```bash
tmux new -s enc      # survives disconnects;  detach: Ctrl-b then d   reattach: tmux attach -t enc

# per-layer probe EER for each encoder (features cached, so re-runs are instant)
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder mert
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder wav2vec2
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder muq
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder encodec
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder xlsr
```
- **CPU** (super, GPU down): ~5 s/clip → ~8 h per SSL encoder. **GPU**: ~50× faster.
- ✅ Paste `data_store/results_stage1.csv` + each per-layer table back to Claude.

---

### Notes
- **Resumable**: per-clip all-layer features cache to
  `data_store/subset_export_round1/features/<encoder>/` (on `$WORK`). Re-runs + probing are instant.
- **When GPU is fixed** (super): reinstall the CUDA torch build matching `nvidia-smi`, re-run same
  commands — much faster.
- **To re-export with different preprocessing** (e.g. 60 s crop): Claude re-runs
  `part2_analysis/export_subset.py` on the Mac (preprocessing lives in Part 2 now); you re-rsync §4.
- **Part layout**: `run_stage1.py` (this Fudan flow) is **self-contained** — it needs only
  `part3_detector/` (encoders/classifiers/eval). `run_stage0.py` and `diagnostics/` now import
  preprocessing/manifest from `part2_analysis/`, so if you run *those* on the server, rsync
  `part2_analysis/` too. §1 above copies only `part3_detector/`, which is all `run_stage1.py` needs.
- **Encoders**: `mert`/`wav2vec2`/`xlsr` = HF SSL (13 layers); `muq` = MuQ music SSL (13 layers,
  needs `muq` pip pkg); `encodec` = neural-codec probe (1 layer). All output `[n_layers, 2*dim]`.
