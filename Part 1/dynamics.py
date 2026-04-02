"""
1_dynamics.py — Dynamics & Loudness Analysis

Panels:
  1. Waveform + RMS envelope + onsets + beats
  2. Crest factor (rolling peak-to-RMS ratio)
  3. Summary statistics text panel

Metrics computed (printed + embedded):
  ── From original file ──
  • RMS Mean (dB)             — average loudness
  • RMS Std (dB)              — loudness variation (low = AI-uniform)
  • Dynamic Range (dB)        — peak minus floor loudness
  • Crest Mean                — average peak/RMS ratio
  • Crest Std                 — variability of crest factor

  ── NEW: added for AI vs human analysis ──
  • RMS Range (dB)            — max minus min RMS (absolute spread)
  • RMS IQR (dB)              — interquartile range of RMS (robust spread)
  • Crest Min                 — minimum crest factor (most compressed moment)
  • Crest < 2.0 (%)           — % of frames with heavy compression
  • Loudness Arc Slope        — linear regression slope of RMS over time
                                (flat = AI-like uniform loudness, non-zero = natural verse/chorus arcs)
  • RMS Autocorrelation lag-1 — temporal smoothness of loudness
                                (very high = unnaturally smooth envelope)

Usage:
    python 1_dynamics.py audio.mp3
    python 1_dynamics.py audio.mp3 -o dynamics.png
"""

import numpy as np
import librosa
from analysis_utils import (
    load_audio, time_axis, make_figure, save_or_show,
    trim_to_shortest, base_argparser,
)


def analyze_dynamics(audio_path: str, sr_target: int = 22050, hop_length: int = 512):
    y, sr, duration, y_harm, y_perc = load_audio(audio_path, sr_target)

    # Beat / onset detection (for waveform overlay)
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    onset_frames = librosa.onset.onset_detect(
        y=y_perc, sr=sr, units="frames",
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.06, wait=10,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Frame times
    n_frames = 1 + len(y) // hop_length
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # RMS
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))

    # Crest factor (rolling peak / RMS)
    peak_env = np.array([
        np.max(np.abs(y[max(0, i * hop_length - 1024): i * hop_length + 1024]))
        for i in range(len(rms))
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 1e-6, peak_env / rms, 0.0)

    crest_valid = crest[crest > 0]
    rms_db_valid = rms_db[rms_db > -60]

    # ── NEW metrics ───────────────────────────────────────────────────────────
    # Loudness arc: linear fit of RMS dB over time → slope
    ft_trim, rms_db_trim = trim_to_shortest(frame_times, rms_db)
    valid_mask = rms_db_trim > -60
    if valid_mask.sum() > 2:
        slope, _ = np.polyfit(ft_trim[valid_mask], rms_db_trim[valid_mask], 1)
    else:
        slope = 0.0

    # RMS autocorrelation at lag-1 (temporal smoothness)
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

    return dict(
        y=y, sr=sr, duration=duration,
        times_samples=np.linspace(0, duration, num=len(y)),
        frame_times=frame_times,
        beat_times=beat_times, onset_times=onset_times,
        rms=rms, crest=crest,
        hop_length=hop_length,
        stats=stats,
    )


def plot_dynamics(data: dict, audio_path: str, output_path: str = None):
    filename = audio_path.split("/")[-1]
    dur   = data["duration"]
    ft    = data["frame_times"]
    rms   = data["rms"]
    crest = data["crest"]
    stats = data["stats"]

    ft, rms, crest = trim_to_shortest(ft, rms, crest)

    fig, axes = make_figure(3, height_per_panel=3.5,
                            title=f"Dynamics & Loudness Analysis — {filename}")
    a0, a1, a2 = axes

    # ── Panel 1: Waveform + RMS + onsets + beats ─────────────────────────────
    a0.plot(data["times_samples"], data["y"], color="#4a90d9", lw=0.3, alpha=0.7)
    rms_scaled = rms / (rms.max() + 1e-9) * np.abs(data["y"]).max()
    a0.fill_between(ft, rms_scaled, -rms_scaled,
                    color="#e8734a", alpha=0.35, label="RMS envelope")
    a0.vlines(data["onset_times"], -1, 1, color="#f5c842", lw=0.6, alpha=0.55,
              label=f"Onsets ({len(data['onset_times'])})")
    a0.vlines(data["beat_times"], -1, 1, color="#7aff7a", lw=0.8, alpha=0.4,
              label=f"Beats ({len(data['beat_times'])})")
    a0.set_ylim(-1.1, 1.1)
    a0.set_ylabel("Amplitude", fontsize=8)
    a0.set_title("Waveform  +  RMS Envelope  +  Onsets  +  Beats", fontsize=9)
    a0.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    time_axis(a0, dur)

    # ── Panel 2: Crest factor ────────────────────────────────────────────────
    a1.plot(ft, crest, color="#ef9a9a", lw=0.8, label="Crest factor (peak/RMS)")
    a1.axhline(stats["crest_mean"], color="#ef9a9a", lw=1.2,
               linestyle="--", alpha=0.6, label=f"Mean = {stats['crest_mean']:.2f}")
    a1.set_ylabel("Crest factor", fontsize=8)
    a1.set_title("Crest Factor — dynamic compression "
                 "(low = brickwall-limited / AI-uniform, high = dynamic / natural)", fontsize=9)
    a1.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    time_axis(a1, dur)

    # ── Panel 3: Stats text ──────────────────────────────────────────────────
    a2.axis("off")
    lines = [
        ("RMS Mean",            f"{stats['rms_mean_db']} dB"),
        ("RMS Std",             f"{stats['rms_std_db']} dB   ← low = uniform loudness (AI trait)"),
        ("Dynamic Range",       f"{stats['dynamic_range_db']} dB"),
        ("RMS Range",           f"{stats['rms_range_db']} dB"),
        ("RMS IQR",             f"{stats['rms_iqr_db']} dB"),
        ("Crest Mean",          f"{stats['crest_mean']}   ← low = heavily compressed"),
        ("Crest Std",           f"{stats['crest_std']}"),
        ("Crest Min",           f"{stats['crest_min']}"),
        ("Crest < 2.0",        f"{stats['crest_below_2_pct']}%   ← high = sustained compression"),
        ("Loudness Arc Slope",  f"{stats['loudness_arc_slope']} dB/s   ← ~0 = flat loudness (AI trait)"),
        ("RMS Autocorr lag-1",  f"{stats['rms_autocorr_lag1']}   ← very high = unnaturally smooth"),
    ]
    col_x = [0.02, 0.22]
    y_pos = 0.95
    a2.text(0.5, 1.0, "Dynamics Summary", ha="center", va="top",
            color="white", fontsize=9, transform=a2.transAxes, weight="bold")
    for label, value in lines:
        a2.text(col_x[0], y_pos, label, color="#aaa", fontsize=8.5,
                transform=a2.transAxes, va="top")
        a2.text(col_x[1], y_pos, value, color="white", fontsize=8.5,
                transform=a2.transAxes, va="top")
        y_pos -= 0.08

    save_or_show(fig, output_path)

    print("\n── Dynamics Stats ──────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<24} {v}")
    print("────────────────────────────────────────────────────────\n")


def main():
    parser = base_argparser("Dynamics & loudness analysis for AI vs. human music research.")
    args = parser.parse_args()
    print(f"[dynamics] Analyzing: {args.audio_path}")
    data = analyze_dynamics(args.audio_path)
    plot_dynamics(data, args.audio_path, output_path=args.output)


if __name__ == "__main__":
    main()