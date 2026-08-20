"""音色/调性家族(10s 线)。公式自 part2_analysis/timbral.py 原样搬迁(2026-08-20 大修)。
HPSS/chroma 改由 ctx 提供(同参数,数值不变);trim 辅助函数本地复制,不再依赖
analysis_utils(那边捆着 matplotlib)。
"""
import numpy as np
import librosa
from scipy.stats import entropy as sp_entropy


def _trim_to_shortest(*arrays):
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


def run(ctx):
    y, sr, hop_length = ctx.y, ctx.SR, ctx.HOP
    duration = ctx.duration
    y_harm, y_perc = ctx.y_harm, ctx.y_perc

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    spec_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=24,
                                        hop_length=hop_length)

    rms_harm = librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=hop_length)[0]

    hp_ratio_mean = float(np.mean(rms_harm) / (np.mean(rms_perc) + 1e-9))

    zcr_std = float(np.std(zcr))
    flat_std = float(np.std(spec_flatness))
    zcr_t, flat_t = _trim_to_shortest(zcr, spec_flatness)
    zcr_flat_corr = float(np.corrcoef(zcr_t, flat_t)[0, 1]) if len(zcr_t) > 2 else 0.0

    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean_abs = float(np.mean(np.abs(mfcc_delta)))

    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    chroma_ent = np.array([sp_entropy(chroma_norm[:, i] + 1e-12) for i in range(chroma.shape[1])])
    chroma_entropy_mean = float(np.mean(chroma_ent))

    active_pc = np.array([
        np.sum(chroma[:, i] > 0.2 * chroma[:, i].max()) for i in range(chroma.shape[1])
    ])
    active_pc_mean = float(np.mean(active_pc))

    dominant_pc = np.argmax(chroma, axis=0)
    changes = np.sum(np.diff(dominant_pc) != 0)
    chord_change_rate = changes / (duration + 1e-9)

    rms_h_t, rms_p_t = _trim_to_shortest(rms_harm, rms_perc)
    with np.errstate(divide="ignore", invalid="ignore"):
        hp_per_frame = np.where(rms_p_t > 1e-9, rms_h_t / rms_p_t, 0.0)
    hp_valid = hp_per_frame[hp_per_frame > 0]

    stats = {
        "zcr_mean":              round(float(np.mean(zcr)), 4),
        "zcr_std":               round(zcr_std, 4),
        "spec_flat_mean":        round(float(np.mean(spec_flatness)), 4),
        "spec_flat_std":         round(flat_std, 4),
        "zcr_flat_corr":         round(zcr_flat_corr, 4),
        "harm_perc_ratio":       round(hp_ratio_mean, 3),
        "hp_ratio_std":          round(float(np.std(hp_valid)), 3) if len(hp_valid) > 0 else 0.0,
        "hp_ratio_min":          round(float(np.min(hp_valid)), 3) if len(hp_valid) > 0 else 0.0,
        "hp_ratio_max":          round(float(np.max(hp_valid)), 3) if len(hp_valid) > 0 else 0.0,
        "mfcc_means":            [round(float(m), 2) for m in mfcc_means],
        "mfcc_stds":             [round(float(s), 2) for s in mfcc_stds],
        "mfcc_delta_mean_abs":   round(mfcc_delta_mean_abs, 3),
        "chroma_entropy_mean":   round(chroma_entropy_mean, 4),
        "active_pc_per_frame":   round(active_pc_mean, 2),
        "chord_change_rate_hz":  round(chord_change_rate, 3),
    }
    return {"stats": stats}
