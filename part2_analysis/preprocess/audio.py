"""
Canonical audio preprocessing — shared by every experiment so codec/loudness/length
confounds are neutralized once, centrally.

load_canonical(): decode -> mono -> resample -> fixed crop -> LUFS loudness-normalize.
"""
import numpy as np
import librosa

try:
    import pyloudnorm as pyln
    _HAVE_PYLN = True
except Exception:
    _HAVE_PYLN = False


def _fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    # pad short clips by tiling then trimming (avoids long digital silence)
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    reps = int(np.ceil(target_len / len(y)))
    return np.tile(y, reps)[:target_len]


def _loudness_normalize(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if not _HAVE_PYLN:
        # fallback: RMS normalize to a fixed level
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        return y * (10 ** (target_lufs / 20.0)) / rms
    try:
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(y)
        if not np.isfinite(loud):
            return y
        return pyln.normalize.loudness(y, loud, target_lufs).astype(np.float32)
    except Exception:
        return y


def load_canonical(path: str, pp: dict) -> np.ndarray:
    """Return a 1-D float32 waveform, preprocessed per the `preprocess` config block."""
    sr = pp["sr"]
    y, _ = librosa.load(path, sr=sr, mono=pp.get("mono", True),
                        offset=pp.get("offset_s", 0.0), duration=pp.get("crop_s", None))
    y = np.asarray(y, dtype=np.float32)
    target_len = int(round(pp["crop_s"] * sr))
    y = _fix_length(y, target_len)
    # peak-guard before loudness metering
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y = y / peak
    y = _loudness_normalize(y, sr, pp.get("loudness_lufs", -23.0))
    # final clip guard
    return np.clip(y, -1.0, 1.0).astype(np.float32)
