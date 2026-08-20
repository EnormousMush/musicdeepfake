"""动态/响度家族(10s 线)。公式自 part2_analysis/dynamics.py 原样搬迁(2026-08-20 大修)。
beat/onset 检测改由 ctx 提供(dynamics 只用于画图,不进 stats,故此处不再要);
frame_times 保留 —— loudness_arc_slope 的线性拟合要用它当 x 轴。
"""
import numpy as np
import librosa


def _trim_to_shortest(*arrays):
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


def run(ctx):
    y, sr, hop_length = ctx.y, ctx.SR, ctx.HOP

    n_frames = 1 + len(y) // hop_length
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))

    peak_env = np.array([
        np.max(np.abs(y[max(0, i * hop_length - 1024): i * hop_length + 1024]))
        for i in range(len(rms))
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 1e-6, peak_env / rms, 0.0)

    crest_valid = crest[crest > 0]
    rms_db_valid = rms_db[rms_db > -60]

    ft_trim, rms_db_trim = _trim_to_shortest(frame_times, rms_db)
    valid_mask = rms_db_trim > -60
    if valid_mask.sum() > 2:
        slope, _ = np.polyfit(ft_trim[valid_mask], rms_db_trim[valid_mask], 1)
    else:
        slope = 0.0

    rms_centered = rms - np.mean(rms)
    if np.std(rms) > 1e-9:
        rms_autocorr = np.corrcoef(rms_centered[:-1], rms_centered[1:])[0, 1]
    else:
        rms_autocorr = 0.0

    stats = {
        "rms_mean_db":        round(float(np.mean(rms_db_valid)), 1),
        "rms_std_db":         round(float(np.std(rms_db_valid)), 2),
        "dynamic_range_db":   round(float(np.max(rms_db_valid) - np.min(rms_db_valid)), 1),
        "rms_range_db":       round(float(np.ptp(rms_db_valid)), 1),
        "rms_iqr_db":         round(float(np.percentile(rms_db_valid, 75)
                                          - np.percentile(rms_db_valid, 25)), 2),
        "crest_mean":         round(float(np.mean(crest_valid)), 2),
        "crest_std":          round(float(np.std(crest_valid)), 2),
        "crest_min":          round(float(np.min(crest_valid)), 2),
        "crest_below_2_pct":  round(float(np.mean(crest_valid < 2.0) * 100), 1),
        "loudness_arc_slope": round(float(slope), 4),
        "rms_autocorr_lag1":  round(float(rms_autocorr), 4),
    }
    return {"stats": stats}
