"""
【2026-08-20 冻结】特征提取已迁往 part2_analysis/features/families/timbral.py(公式原样,对拍逐位一致)。本文件保留:历史口径参照 + plot_timbral 单曲诊断。

3_timbral.py — Timbral & Tonal Analysis

Panels:
  1. Chromagram (pitch-class energy from harmonic component)
  2. MFCCs (20 coefficients — timbral texture over time)
  3. ZCR + Spectral Flatness (noisiness / tonality)
  4. Harmonic vs. Percussive Energy (HPSS)
  5. Summary statistics text panel

Metrics computed (printed + embedded):
  ── From original file ──
  • ZCR Mean                  — noisiness proxy
  • Spectral Flatness Mean    — tonal vs noise-like
  • Harm/Perc Ratio           — mean harmonic RMS / mean percussive RMS

  ── NEW: added for AI vs human analysis ──
  • ZCR Std                   — variation of zero-crossing rate
  • Flatness Std              — variation of spectral flatness
  • ZCR-Flatness Correlation  — how tightly these two noise measures track each other
  • MFCC Means (20)           — per-coefficient averages (broad timbral fingerprint)
  • MFCC Stds  (20)           — per-coefficient variation (low in higher MFCCs → less micro-detail, AI trait)
  • MFCC Delta Mean Abs       — average magnitude of MFCC first derivative
                                (low = sluggish timbral movement → AI trait)
  • Chroma Entropy Mean       — pitch-class entropy per frame, averaged
                                (low = repetitive harmony → AI trait)
  • Active Pitch Classes/Frame— average # of pitch classes with significant energy
  • Chord Change Rate         — how often the dominant pitch class changes (per second)
  • H/P Ratio Std             — variation of harmonic-to-percussive ratio over time
                                (low = unnaturally consistent balance → AI trait)
  • H/P Ratio Min / Max       — extremes of the harmonic-percussive balance

Usage:
    python 3_timbral.py audio.mp3
    python 3_timbral.py audio.mp3 -o timbral.png
"""

import numpy as np
import librosa
import librosa.display
from scipy.stats import entropy as sp_entropy
from analysis_utils import (
    load_audio, time_axis, make_figure, save_or_show,
    trim_to_shortest, base_argparser,
)
import matplotlib.pyplot as plt


def analyze_timbral(audio_path: str, sr_target: int = 22050, hop_length: int = 512):
    y, sr, duration, y_harm, y_perc = load_audio(audio_path, sr_target)

    n_frames = 1 + len(y) // hop_length
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # Core features
    zcr           = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    spec_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    chroma        = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=24,
                                                hop_length=hop_length)

    # Harmonic / percussive RMS
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=hop_length)[0]

    hp_ratio_mean = float(np.mean(rms_harm) / (np.mean(rms_perc) + 1e-9))

    # ── NEW metrics ───────────────────────────────────────────────────────────
    # ZCR / flatness stats
    zcr_std       = float(np.std(zcr))
    flat_std      = float(np.std(spec_flatness))
    zcr_t, flat_t = trim_to_shortest(zcr, spec_flatness)
    zcr_flat_corr = float(np.corrcoef(zcr_t, flat_t)[0, 1]) if len(zcr_t) > 2 else 0.0

    # MFCC statistics
    mfcc_means = np.mean(mfcc, axis=1)     # (20,)
    mfcc_stds  = np.std(mfcc, axis=1)      # (20,)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean_abs = float(np.mean(np.abs(mfcc_delta)))

    # Chroma-based harmonic richness
    # Per-frame entropy of the 12 pitch classes
    chroma_norm = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    chroma_ent  = np.array([sp_entropy(chroma_norm[:, i] + 1e-12) for i in range(chroma.shape[1])])
    chroma_entropy_mean = float(np.mean(chroma_ent))

    # Active pitch classes: count per frame where energy > 20% of frame max
    active_pc = np.array([
        np.sum(chroma[:, i] > 0.2 * chroma[:, i].max()) for i in range(chroma.shape[1])
    ])
    active_pc_mean = float(np.mean(active_pc))

    # Chord change rate: count how often dominant pitch class changes
    dominant_pc = np.argmax(chroma, axis=0)
    changes = np.sum(np.diff(dominant_pc) != 0)
    chord_change_rate = changes / (duration + 1e-9)

    # H/P ratio per frame
    rms_h_t, rms_p_t = trim_to_shortest(rms_harm, rms_perc)
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

    return dict(
        sr=sr, duration=duration,
        frame_times=frame_times,
        zcr=zcr, spec_flatness=spec_flatness,
        mfcc=mfcc, chroma=chroma,
        rms_harm=rms_harm, rms_perc=rms_perc,
        hop_length=hop_length,
        stats=stats,
    )


def plot_timbral(data: dict, audio_path: str, output_path: str = None):
    filename = audio_path.split("/")[-1]
    sr  = data["sr"]
    hop = data["hop_length"]
    dur = data["duration"]
    ft  = data["frame_times"]
    stats = data["stats"]

    zcr = data["zcr"]
    sf  = data["spec_flatness"]
    rms_harm = data["rms_harm"]
    rms_perc = data["rms_perc"]

    ft, zcr, sf, rms_harm, rms_perc = trim_to_shortest(ft, zcr, sf, rms_harm, rms_perc)

    fig, axes = make_figure(5, height_per_panel=3.2,
                            title=f"Timbral & Tonal Analysis — {filename}")
    a0, a1, a2, a3, a4 = axes

    # ── Panel 1: Chromagram ──────────────────────────────────────────────────
    img = librosa.display.specshow(data["chroma"], sr=sr, hop_length=hop,
                                   x_axis="time", y_axis="chroma", cmap="coolwarm", ax=a0)
    fig.colorbar(img, ax=a0, pad=0.01).ax.tick_params(labelsize=7, colors="#aaa")
    a0.set_title("Chromagram — pitch-class energy (harmonic component)", fontsize=9)
    a0.set_ylabel("Pitch", fontsize=8)

    # ── Panel 2: MFCCs ───────────────────────────────────────────────────────
    img = librosa.display.specshow(data["mfcc"], sr=sr, hop_length=hop,
                                   x_axis="time", cmap="RdBu_r", ax=a1)
    fig.colorbar(img, ax=a1, pad=0.01).ax.tick_params(labelsize=7, colors="#aaa")
    a1.set_title("MFCCs — timbral texture (20 coefficients)", fontsize=9)
    a1.set_ylabel("MFCC #", fontsize=8)

    # ── Panel 3: ZCR + Spectral flatness ─────────────────────────────────────
    ax2 = a2.twinx()
    a2.plot(ft, zcr, color="#aed581", lw=0.8, label="ZCR")
    ax2.plot(ft, sf, color="#ff8a65", lw=0.8, alpha=0.8, label="Flatness")
    a2.set_ylabel("ZCR", fontsize=8, color="#aed581")
    ax2.set_ylabel("Flatness", fontsize=8, color="#ff8a65")
    ax2.tick_params(colors="#ff8a65", labelsize=8)
    a2.set_title("Zero-Crossing Rate  +  Spectral Flatness — noise vs. tonality", fontsize=9)
    l1, lb1 = a2.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    a2.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7,
              framealpha=0.3, labelcolor="white")
    time_axis(a2, dur)

    # ── Panel 4: Harmonic vs Percussive energy ───────────────────────────────
    a3.fill_between(ft, rms_harm, color="#ce93d8", alpha=0.7, label="Harmonic RMS")
    a3.fill_between(ft, rms_perc, color="#80cbc4", alpha=0.5, label="Percussive RMS")
    a3.set_ylabel("RMS", fontsize=8)
    a3.set_title("Harmonic vs. Percussive Energy (HPSS)", fontsize=9)
    a3.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    time_axis(a3, dur)

    # ── Panel 5: Stats text ──────────────────────────────────────────────────
    a4.axis("off")
    lines = [
        ("ZCR Mean",             f"{stats['zcr_mean']}"),
        ("ZCR Std",              f"{stats['zcr_std']}"),
        ("Spectral Flatness",    f"{stats['spec_flat_mean']}   ← high=noise-like, low=tonal"),
        ("Flatness Std",         f"{stats['spec_flat_std']}"),
        ("ZCR–Flatness Corr",    f"{stats['zcr_flat_corr']}"),
        ("Harm/Perc Ratio",      f"{stats['harm_perc_ratio']}"),
        ("H/P Ratio Std",        f"{stats['hp_ratio_std']}   ← low = unnaturally consistent (AI)"),
        ("H/P Ratio Range",      f"{stats['hp_ratio_min']} – {stats['hp_ratio_max']}"),
        ("MFCC Δ Mean |Δ|",      f"{stats['mfcc_delta_mean_abs']}   ← low = sluggish timbre (AI)"),
        ("Chroma Entropy",       f"{stats['chroma_entropy_mean']}   ← low = repetitive harmony (AI)"),
        ("Active PCs / frame",   f"{stats['active_pc_per_frame']}"),
        ("Chord Change Rate",    f"{stats['chord_change_rate_hz']} /s"),
    ]
    col_x = [0.02, 0.22]
    y_pos = 0.95
    a4.text(0.5, 1.0, "Timbral & Tonal Summary", ha="center", va="top",
            color="white", fontsize=9, transform=a4.transAxes, weight="bold")
    for label, value in lines:
        a4.text(col_x[0], y_pos, label, color="#aaa", fontsize=8.5,
                transform=a4.transAxes, va="top")
        a4.text(col_x[1], y_pos, value, color="white", fontsize=8.5,
                transform=a4.transAxes, va="top")
        y_pos -= 0.07

    save_or_show(fig, output_path)

    print("\n── Timbral & Tonal Stats ───────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("────────────────────────────────────────────────────────\n")


def main():
    parser = base_argparser("Timbral & tonal analysis for AI vs. human music research.")
    args = parser.parse_args()
    print(f"[timbral] Analyzing: {args.audio_path}")
    data = analyze_timbral(args.audio_path)
    plot_timbral(data, args.audio_path, output_path=args.output)


if __name__ == "__main__":
    main()