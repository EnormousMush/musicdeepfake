# Fudan server run — Round-1 pipeline validation (offline playbook)

**Read this whole file first.** You run it during ONE Fudan-VPN session, during which you
**cannot reach Claude** — so this file is your reference. Each step says **[MAC]** (run in your
Mac terminal) or **[SERVER]** (run inside the ssh session). Checkpoints marked ✅ tell you what
"good" looks like; if a ✅ fails, **save the output and come back to Claude** (don't push past it).

## Facts this playbook is built on (from recon)
- Server: `FUDAN_USER@FUDAN_HOST`, Ubuntu 20.04, **2× RTX 3060 12 GB**, driver **470 → CUDA 11.4 max**, Python **3.8**, no conda.
- **Root disk `/` is 100% full (3.4 G)** → everything goes on `/home/FUDAN_USER/mnt` (**688 G free**).
- Server can reach **PyPI / Tsinghua**, but **HuggingFace + GitHub are BLOCKED** → we ship the model
  cache and run HuggingFace **offline**.
- Old driver → **old pinned stack**: `torch 1.12.1+cu113` + `transformers 4.35.2` (NOT the Mac's torch 2.8).
- Use **GPU 1** (`CUDA_VISIBLE_DEVICES=1`); GPU 0 is ~full with someone else's job. Shared machine — be polite.

Goal: run all encoders' per-layer probes on the round-1 set to **finalize the pipeline** (encoder/layer
choices), not just prove it runs. `wav2vec2`/`encodec`/`xlsr` are low-risk; `mert`/`muq` may need the old
stack to cooperate — run them last, and if one fails, note it and keep going.

---

## Phase 0 — before you switch VPN [MAC]
```bash
ls -d "/Volumes/Seagate /frank-suno-round1"   # drive mounted + staging present?
```
✅ prints the path. If not, mount the Seagate drive first (volume name has a trailing space).

Then **switch to the Fudan VPN.** From here you can't reach Claude until you switch back.

Reusable shell vars — paste this in EVERY new Mac terminal you open this session:
```bash
SRV=FUDAN_USER@FUDAN_HOST
WORK=/home/FUDAN_USER/mnt/frank-suno/detector
STAGE="/Volumes/Seagate /frank-suno-round1"
```

---

## Phase 1 — push code to the server [MAC]
```bash
cd ~/Developer/frank-suno-backup/part3_detector
rsync -av --exclude .venv --exclude .git --exclude data_store --exclude .hfcache \
  ./ "$SRV:$WORK/"
```
(enter the server password when prompted — it won't display as you type; first connect also asks yes/no → `yes`.)
✅ `ssh $SRV "ls $WORK"` shows `run_stage1.py`, `encoders/`, `check_server.py`.

---

## Phase 2 — set up the env on the server [SERVER]
SSH in and run this whole block. **The `export`s must be set or pip will fill the full root disk.**
```bash
ssh $SRV        # then, inside the server:

export WORK=/home/FUDAN_USER/mnt/frank-suno/detector
mkdir -p "$WORK"/{.tmp,.pipcache,.hfcache}
export TMPDIR="$WORK/.tmp"
export PIP_CACHE_DIR="$WORK/.pipcache"
export HF_HOME="$WORK/.hfcache"
cd "$WORK"

# 1) make a venv on the big disk (fallback to virtualenv if python3-venv is missing)
python3 -m venv .venv 2>/dev/null && echo "venv OK" || {
  echo "no venv module -> using virtualenv";
  PYTHONUSERBASE="$WORK/.pybase" pip3 install --user -i https://pypi.tuna.tsinghua.edu.cn/simple virtualenv
  "$WORK/.pybase/bin/virtualenv" -p python3 .venv
}
source .venv/bin/activate
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) pin torch to the cu113 build that matches driver 470 (from pytorch.org; slow in CN but usually works)
pip install --timeout 180 --retries 5 \
  torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  --index-url https://download.pytorch.org/whl/cu113
```
✅ **GPU CHECKPOINT** — run this and confirm `True` + a 3060:
```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```
If it prints `... True NVIDIA GeForce RTX 3060` → continue.
If `False` or the torch install failed/hung → **stop, copy the output, come back to Claude.**

```bash
# 3) the rest of the deps, pinned so pip can't upgrade torch/numpy — from Tsinghua (fast)
cat > "$WORK/constraints.txt" <<'C'
torch==1.12.1+cu113
torchvision==0.13.1+cu113
torchaudio==0.12.1+cu113
numpy==1.24.4
C
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -c "$WORK/constraints.txt" \
  "transformers==4.35.2" "numpy==1.24.4" soundfile librosa pyloudnorm scikit-learn pyyaml nnAudio

# 4) MuQ (optional / riskiest on old torch) — constrained so it can't touch torch. OK if it fails.
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -c "$WORK/constraints.txt" muq \
  && echo "muq OK" || echo "muq install FAILED — skip the muq encoder, the others are fine"
```
✅ `python -c "import transformers, soundfile, librosa, sklearn; print('deps ok')"` prints `deps ok`.

---

## Phase 3 — ship data + model cache from the Mac [MAC]
Open a **Mac terminal** (re-paste the `SRV/WORK/STAGE` vars from Phase 0). Big transfers (~7 GB) —
`-P` shows progress and resumes if interrupted.
```bash
# model cache (2.4 G) — REQUIRED (server can't download from HuggingFace)
rsync -avP "$STAGE/hfcache/" "$SRV:$WORK/.hfcache/"

# data (4.6 G)
rsync -avP "$STAGE/subset_export_round1" "$SRV:$WORK/data_store/"
```
✅ `ssh $SRV "ls $WORK/data_store/subset_export_round1/audio | wc -l"` → **6000**.
✅ `ssh $SRV "ls $WORK/.hfcache/hub"` shows the 4 `models--*` dirs.

---

## Phase 4 — verify on GPU (tiny) [SERVER]
Back in the ssh session (re-set the exports if it's a new shell). **Offline flags matter** — without
them HuggingFace calls hang trying to reach the blocked network.
```bash
export WORK=/home/FUDAN_USER/mnt/frank-suno/detector
export TMPDIR="$WORK/.tmp" PIP_CACHE_DIR="$WORK/.pipcache" HF_HOME="$WORK/.hfcache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # HF is blocked → force cache-only
export CUDA_VISIBLE_DEVICES=1                       # use GPU 1 (GPU 0 is busy)
cd "$WORK" && source .venv/bin/activate

python check_server.py --encoder wav2vec2          # safest encoder first (no remote code)
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder wav2vec2 --limit 50
```
✅ `check_server.py` prints `cuda available: True` and a feature shape.
✅ `run_stage1.py` prints a per-layer EER table (50-clip numbers are meaningless — only that it runs on GPU).
If this works, the pipeline is alive on the Fudan GPU.

---

## Phase 5 — full per-layer probes, all encoders (tmux) [SERVER]
GPU makes each encoder ~10–20 min (not hours). Use tmux so a disconnect doesn't kill it.
```bash
tmux new -s enc        # detach: Ctrl-b then d   |   reattach: tmux attach -t enc
# (re-run the Phase-4 exports inside tmux if it's a fresh shell)

# safest first, riskiest last — so a late failure still leaves you results
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder wav2vec2
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder encodec
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder xlsr
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder mert     # remote code — may warn/fail
python run_stage1.py --data-dir data_store/subset_export_round1 --encoder muq      # newest — may fail on old torch
```
If `mert` or `muq` errors, that's OK — copy the error, keep the others. Features cache to
`data_store/subset_export_round1/features/<encoder>/`, so nothing is wasted on a re-run.

---

## Phase 6 — collect results, then come back [SERVER → MAC]
```bash
cat "$WORK/data_store/results_stage1.csv"     # [SERVER] — copy this
```
Also copy each encoder's **per-layer EER table** from the scroll-back (or re-run one; it's instant from cache).

Then **switch VPN back to your normal network** and paste to Claude:
1. `results_stage1.csv` contents,
2. each per-layer table,
3. any error from `mert`/`muq` (if they failed).

That's the round-1 result we use to **finalize the encoder + layer choice**.

---

### If something breaks and you're stuck offline
- **torch says CUDA False / install failed** → likely `download.pytorch.org` was unreachable or slow.
  Save the output; Claude will give a China mirror. (Don't try random `sudo`.)
- **root disk fills / "No space left"** → an `export` (TMPDIR/PIP_CACHE_DIR/HF_HOME) wasn't set to `$WORK`.
  Re-set all of them and retry.
- **mert/muq crash** → expected risk on the old stack; skip them, keep `wav2vec2/encodec/xlsr`.
- Everything is **resumable** — re-running any phase is safe (rsync skips existing, features are cached).
