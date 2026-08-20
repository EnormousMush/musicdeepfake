"""节奏/时值家族(10s 线)。公式自 part2_analysis/rhythm.py 原样搬迁(2026-08-20 大修)。
beat/onset/onset_env 改由 ctx 提供(同参数同分量,数值不变)。
注:旧版把原始 ibis 数组返回在顶层,被 extract 的展开器泄成 ibis_00..N 事故列
(2026-08-20 冗余审计发现)—— 新版只返回设计过的 stats,事故列就此绝版。
"""
import numpy as np


def run(ctx):
    duration = ctx.duration
    tempo = ctx.tempo
    beat_times = ctx.beat_times
    onset_frames = ctx.onset_frames
    onset_times = ctx.onset_times
    onset_env = ctx.onset_env_perc
    sr, hop_length = ctx.SR, ctx.HOP

    ibis = np.diff(beat_times) if len(beat_times) > 1 else np.array([0.0])
    ibi_mean = float(np.mean(ibis))
    ibi_std = float(np.std(ibis))
    ibi_cv = ibi_std / (ibi_mean + 1e-9)

    onset_density = len(onset_times) / (duration + 1e-9)

    os_mean = float(np.mean(onset_env))
    os_std = float(np.std(onset_env))

    onset_peak_vals = onset_env[onset_frames[onset_frames < len(onset_env)]] \
        if len(onset_frames) > 0 else np.array([0.0])
    os_peak_std = float(np.std(onset_peak_vals))

    if len(beat_times) > 0 and len(onset_times) > 0:
        beat_tolerance = 0.5 * ibi_mean if ibi_mean > 0 else 0.1
        on_beat = 0
        for ot in onset_times:
            if np.min(np.abs(ot - beat_times)) <= beat_tolerance * 0.25:
                on_beat += 1
        syncopation = 1.0 - (on_beat / len(onset_times))
    else:
        syncopation = 0.0

    if len(ibis) > 4:
        win = min(8, len(ibis))
        windowed_tempos = 60.0 / np.array([
            np.mean(ibis[i:i + win]) for i in range(len(ibis) - win + 1)
        ])
        tempo_stability = 1.0 - (float(np.std(windowed_tempos)) /
                                 (float(np.mean(windowed_tempos)) + 1e-9))
    else:
        tempo_stability = 1.0

    beat_period_frames = int(round(60.0 / (tempo + 1e-9) * sr / hop_length))
    if beat_period_frames > 0 and beat_period_frames < len(onset_env) // 2:
        oe_centered = onset_env - np.mean(onset_env)
        norm = np.sqrt(np.sum(oe_centered ** 2))
        if norm > 1e-9:
            groove_corr = float(np.sum(
                oe_centered[:-beat_period_frames] * oe_centered[beat_period_frames:]
            ) / (norm ** 2) * len(oe_centered))
        else:
            groove_corr = 0.0
    else:
        groove_corr = 0.0

    stats = {
        "tempo_bpm":            round(tempo, 1),
        "num_onsets":           len(onset_times),
        "onset_density_per_s":  round(onset_density, 2),
        "ibi_mean_s":           round(ibi_mean, 4),
        "ibi_std_s":            round(ibi_std, 4),
        "ibi_cv":               round(ibi_cv, 4),
        "beat_regularity":      round(1.0 - ibi_cv, 4),
        "onset_str_mean":       round(os_mean, 3),
        "onset_str_std":        round(os_std, 3),
        "onset_peak_std":       round(os_peak_std, 3),
        "syncopation_index":    round(syncopation, 4),
        "tempo_stability":      round(tempo_stability, 4),
        "groove_consistency":   round(groove_corr, 4),
    }
    return {"stats": stats}
