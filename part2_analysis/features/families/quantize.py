"""量化程度家族(10s 线)。公式自 part2_analysis/quantize_deg.py 原样搬迁(2026-08-20 大修)。
旧版自己 load + HPSS + beat + onset(与 ctx 完全同参数)—— 全部改由 ctx 提供。
输出列名与历史 CSV 一致:tempo / quantization_score / mean_dev_ms / std_dev_ms /
max_dev_ms / swing_pct / num_onsets / subdivision(tempo、num_onsets 与 rhythm 家族的
stats.tempo_bpm、stats.num_onsets 是历史双胞胎列,保留以保对拍)。
"""
import numpy as np


def _build_local_grid(beat_times: np.ndarray, subdivisions: int) -> np.ndarray:
    grid = []
    for i in range(len(beat_times) - 1):
        beat_dur = beat_times[i + 1] - beat_times[i]
        for s in range(subdivisions):
            grid.append(beat_times[i] + s * beat_dur / subdivisions)
    if len(beat_times) >= 2:
        last_dur = beat_times[-1] - beat_times[-2]
        for s in range(subdivisions):
            grid.append(beat_times[-1] + s * last_dur / subdivisions)
    return np.array(grid)


def _swing_ratio(beat_times: np.ndarray, onset_times: np.ndarray) -> float:
    ratios = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_dur = beat_times[i + 1] - beat_times[i]
        off_beat_onsets = onset_times[
            (onset_times > beat_start + beat_dur * 0.3) &
            (onset_times < beat_start + beat_dur * 0.8)
        ]
        if len(off_beat_onsets) > 0:
            off_beat = off_beat_onsets[0]
            ratios.append((off_beat - beat_start) / beat_dur)
    return float(np.mean(ratios)) if ratios else 0.5


def run(ctx, subdivision: int = 16):
    tempo = ctx.tempo
    beat_times = ctx.beat_times

    if len(beat_times) < 2:
        return {"error": "Not enough beats detected to analyze quantization."}

    onset_times = ctx.onset_times
    onset_times = onset_times[
        (onset_times >= beat_times[0]) & (onset_times <= beat_times[-1])
    ]
    if len(onset_times) == 0:
        return {"error": "No onsets detected."}

    subs_per_beat = max(1, subdivision // 4)
    grid = _build_local_grid(beat_times, subs_per_beat)

    deviations_sec = np.array([
        onset - grid[np.argmin(np.abs(grid - onset))]
        for onset in onset_times
    ])
    abs_devs = np.abs(deviations_sec)

    beat_dur_avg = float(np.mean(np.diff(beat_times)))
    grid_cell = beat_dur_avg / subs_per_beat
    max_possible_dev = grid_cell / 2.0

    mean_dev = float(np.mean(abs_devs))
    std_dev = float(np.std(deviations_sec))
    max_dev = float(np.max(abs_devs))

    quant_score = max(0.0, 100.0 * (1.0 - mean_dev / max_possible_dev))
    swing_pct = _swing_ratio(beat_times, onset_times) * 100.0

    return {
        "tempo":              round(tempo, 1),
        "quantization_score": round(quant_score, 1),
        "mean_dev_ms":        round(mean_dev * 1000, 2),
        "std_dev_ms":         round(std_dev * 1000, 2),
        "max_dev_ms":         round(max_dev * 1000, 2),
        "swing_pct":          round(swing_pct, 1),
        "num_onsets":         len(onset_times),
        "subdivision":        subdivision,
    }
