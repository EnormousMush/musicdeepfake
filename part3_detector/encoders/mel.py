"""
Mel-spectrogram encoder (Stage 0, CPU-friendly, zero heavy deps).

Produces a fixed-length pooled feature vector: log-mel -> temporal {mean, std}.
This is the cheap "measuring stick" encoder. Real SSL encoders (MERT, wav2vec2)
plug into the same interface: encode(waveform, sr, cfg) -> 1-D np.ndarray.
"""
import numpy as np
import librosa

NAME = "mel"


def encode(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    n_mels = cfg.get("n_mels", 128)
    pooling = cfg.get("pooling", ["mean", "std"])
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel + 1e-10)          # [n_mels, T]
    parts = []
    if "mean" in pooling:
        parts.append(logmel.mean(axis=1))
    if "std" in pooling:
        parts.append(logmel.std(axis=1))
    return np.concatenate(parts).astype(np.float32)     # [n_mels * len(pooling)]
