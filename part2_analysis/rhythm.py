"""
【2026-08-20 冻结】特征提取已迁往 part2_analysis/features/families/rhythm.py(公式原样,对拍逐位一致;旧版顶层 ibis 数组曾泄成 CSV 事故列,新版已绝版)。本文件保留:历史口径参照 + plot_rhythm 单曲诊断。

4_rhythm.py — Rhythm & Timing Analysis

Panels:
  1. Onset Strength Envelope + Beat Grid
  2. Inter-Beat Interval (IBI) deviation plot
  3. Summary statistics text panel

Metrics computed (printed + embedded):
  ── From original file ──
  • Tempo (BPM)              — estimated tempo
  • # of Onsets               — total onset events detected

  ── NEW: added for AI vs human analysis ──
  • IBI Mean (s)              — mean inter-beat interval
  • IBI Std (s)               — standard deviation of inter-beat intervals
                                (near-zero = metronomic / AI-locked)
  • IBI CV                    — coefficient of variation (std/mean) of IBIs
                                (normalised regularity; <0.02 is suspiciously perfect)
  • Beat Regularity Score     — 1 – IBI_CV  (higher = more regular; >0.98 → AI-like)
  • Onset Density (/s)        — onsets per second
  • Onset Strength Mean       — average onset strength (rhythmic energy)
  • Onset Strength Std        — variation of onset strength
  • Onset Strength Peak Std   — std of just the peak onset strengths (beat accents)
                                (low = uniform accents → AI trait)
  • Syncopation Index         — fraction of onsets that do NOT fall on a beat
                                (low = everything lands on-grid → AI trait)
  • Tempo Stability           — 1 − (tempo_std / tempo_mean) from windowed tempo estimate
  • Groove Consistency        — autocorrelation of onset strength at beat period
                                (very high + no variation = mechanical groove)

Usage:
    python 4_rhythm.py audio.mp3
    python 4_rhythm.py audio.mp3 -o rhythm.png
"""

import numpy as np
import librosa
from analysis_utils import (
    load_audio, time_axis, make_figure, save_or_show,
    base_argparser,
)


def analyze_rhythm(audio_path: str, sr_target: int = 22050, hop_length: int = 512):
    y, sr, duration, _, y_perc = load_audio(audio_path, sr_target)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Onsets
    onset_frames = librosa.onset.onset_detect(
        y=y_perc, sr=sr, units="frames",
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.06, wait=10,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_env   = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)
    oe_times    = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)

    # ── NEW metrics ───────────────────────────────────────────────────────────
    # Inter-beat intervals
    ibis = np.diff(beat_times) if len(beat_times) > 1 else np.array([0.0])
    ibi_mean = float(np.mean(ibis))
    ibi_std  = float(np.std(ibis))
    ibi_cv   = ibi_std / (ibi_mean + 1e-9)

    # Onset density
    onset_density = len(onset_times) / (duration + 1e-9)

    # Onset strength stats
    os_mean = float(np.mean(onset_env))
    os_std  = float(np.std(onset_env))

    # Peak onset strengths (at detected onset frames)
    onset_peak_vals = onset_env[onset_frames[onset_frames < len(onset_env)]] \
        if len(onset_frames) > 0 else np.array([0.0])
    os_peak_std = float(np.std(onset_peak_vals))

    # Syncopation: fraction of onsets not within 1 frame of a beat
    if len(beat_times) > 0 and len(onset_times) > 0:
        beat_tolerance = 0.5 * ibi_mean if ibi_mean > 0 else 0.1
        on_beat = 0
        for ot in onset_times:
            if np.min(np.abs(ot - beat_times)) <= beat_tolerance * 0.25:
                on_beat += 1
        syncopation = 1.0 - (on_beat / len(onset_times))
    else:
        syncopation = 0.0

    # Tempo stability from windowed tempo
    if len(ibis) > 4:
        win = min(8, len(ibis))
        windowed_tempos = 60.0 / np.array([
            np.mean(ibis[i:i + win]) for i in range(len(ibis) - win + 1)
        ])
        tempo_stability = 1.0 - (float(np.std(windowed_tempos)) /
                                  (float(np.mean(windowed_tempos)) + 1e-9))
    else:
        tempo_stability = 1.0

    # Groove consistency: autocorrelation of onset_env at lag = beat period in frames
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

    return dict(
        sr=sr, duration=duration,
        beat_times=beat_times, onset_times=onset_times,
        onset_env=onset_env, oe_times=oe_times,
        ibis=ibis,
        hop_length=hop_length,
        stats=stats,
    )


def plot_rhythm(data: dict, audio_path: str, output_path: str = None):
    filename = audio_path.split("/")[-1]
    dur   = data["duration"]
    stats = data["stats"]

    fig, axes = make_figure(3, height_per_panel=3.5,
                            title=f"Rhythm & Timing Analysis — {filename}")
    a0, a1, a2 = axes

    # ── Panel 1: Onset strength + beat markers ───────────────────────────────
    a0.plot(data["oe_times"], data["onset_env"], color="#ffcc02", lw=0.8,
            label="Onset strength")
    a0.vlines(data["beat_times"], 0, data["onset_env"].max(),
              color="#7aff7a", lw=0.9, alpha=0.5, label="Beats")
    a0.set_ylabel("Strength", fontsize=8)
    a0.set_title("Onset Strength Envelope + Beat Grid — rhythmic regularity", fontsize=9)
    a0.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    time_axis(a0, dur)

    # ── Panel 2: IBI deviation plot ──────────────────────────────────────────
    ibis = data["ibis"]
    if len(ibis) > 1:
        expected_ibi = np.median(ibis)
        ibi_deviation_ms = (ibis - expected_ibi) * 1000  # ms
        a1.bar(np.arange(len(ibi_deviation_ms)), ibi_deviation_ms,
               color="#4fc3f7", alpha=0.7, width=0.8)
        a1.axhline(0, color="#888", lw=0.8, linestyle="--")
        a1.set_xlabel("Beat index", fontsize=8)
        a1.set_ylabel("Deviation (ms)", fontsize=8)
        a1.set_title("Inter-Beat Interval Deviation from Median "
                      "(flat bars = metronomic / AI-locked)", fontsize=9)
    else:
        a1.text(0.5, 0.5, "Not enough beats detected", ha="center", va="center",
                color="#aaa", fontsize=10, transform=a1.transAxes)
        a1.set_title("Inter-Beat Interval Deviation", fontsize=9)

    # ── Panel 3: Stats text ──────────────────────────────────────────────────
    a2.axis("off")
    lines = [
        ("Tempo",               f"{stats['tempo_bpm']} BPM"),
        ("Onsets",              str(stats['num_onsets'])),
        ("Onset Density",       f"{stats['onset_density_per_s']} /s"),
        ("IBI Mean",            f"{stats['ibi_mean_s']} s"),
        ("IBI Std",             f"{stats['ibi_std_s']} s   ← ~0 = metronomic (AI trait)"),
        ("IBI CV",              f"{stats['ibi_cv']}   ← <0.02 = suspiciously perfect"),
        ("Beat Regularity",     f"{stats['beat_regularity']}   ← >0.98 = AI-like"),
        ("Onset Strength Mean", f"{stats['onset_str_mean']}"),
        ("Onset Strength Std",  f"{stats['onset_str_std']}"),
        ("Onset Peak Std",      f"{stats['onset_peak_std']}   ← low = uniform accents (AI)"),
        ("Syncopation Index",   f"{stats['syncopation_index']}   ← low = on-grid only (AI)"),
        ("Tempo Stability",     f"{stats['tempo_stability']}"),
        ("Groove Consistency",  f"{stats['groove_consistency']}"),
    ]
    col_x = [0.02, 0.22]
    y_pos = 0.95
    a2.text(0.5, 1.0, "Rhythm Summary", ha="center", va="top",
            color="white", fontsize=9, transform=a2.transAxes, weight="bold")
    for label, value in lines:
        a2.text(col_x[0], y_pos, label, color="#aaa", fontsize=8.5,
                transform=a2.transAxes, va="top")
        a2.text(col_x[1], y_pos, value, color="white", fontsize=8.5,
                transform=a2.transAxes, va="top")
        y_pos -= 0.07

    save_or_show(fig, output_path)

    print("\n── Rhythm Stats ───────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print("────────────────────────────────────────────────────────\n")


def main():
    parser = base_argparser("Rhythm & timing analysis for AI vs. human music research.")
    args = parser.parse_args()
    print(f"[rhythm] Analyzing: {args.audio_path}")
    data = analyze_rhythm(args.audio_path)
    plot_rhythm(data, args.audio_path, output_path=args.output)


if __name__ == "__main__":
    main()