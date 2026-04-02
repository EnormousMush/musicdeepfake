"""
2_spectral.py — Spectral & Brightness Analysis

Panels:
  1. STFT Spectrogram (log frequency, dB)
  2. Mel Spectrogram (perceptual frequency scale)
  3. Spectral Centroid + Rolloff + Bandwidth over time
  4. Summary statistics text panel

Metrics computed (printed + embedded):
  ── From original file ──
  • Centroid Mean (Hz)        — average brightness

  ── NEW: added for AI vs human analysis ──
  • Centroid Std (Hz)         — brightness variation over time
                                (low = static / AI-like timbral profile)
  • Centroid-to-Rolloff Ratio — how concentrated energy is below the rolloff
  • Rolloff Mean (Hz)         — high-frequency cutoff point
  • Rolloff Std (Hz)          — variation of HF cutoff
  • Bandwidth Mean (Hz)       — average spectral spread
  • Bandwidth Std (Hz)        — variation of spread
  • HF Energy Ratio           — fraction of STFT energy above 8 kHz
                                (AI codecs sometimes cut off sharply)
  • Spectral Contrast (per-band mean) — difference between peaks and valleys
                                in each sub-band (less contrast → flatter / synthetic)
  • Per-Mel-Band Energy Std   — temporal variance averaged across mel bands
                                (low = static texture / AI trait)

Usage:
    python 2_spectral.py audio.mp3
    python 2_spectral.py audio.mp3 -o spectral.png
"""

import numpy as np
import librosa
import librosa.display
from analysis_utils import (
    load_audio, time_axis, make_figure, save_or_show,
    trim_to_shortest, base_argparser,
)
import matplotlib.pyplot as plt


def analyze_spectral(audio_path: str, sr_target: int = 22050, hop_length: int = 512):
    y, sr, duration, _, _ = load_audio(audio_path, sr_target)

    n_frames = 1 + len(y) // hop_length
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # STFT spectrogram
    D = librosa.stft(y, hop_length=hop_length)
    S_mag = np.abs(D)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Spectral features
    spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length,
                                                       roll_percent=0.85)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    # ── NEW metrics ───────────────────────────────────────────────────────────
    # Spectral contrast (7 sub-bands by default)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    contrast_band_means = np.mean(spec_contrast, axis=1)  # shape (7,)

    # HF energy ratio (fraction of energy above 8 kHz)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = freqs >= 8000
    hf_energy = np.sum(S_mag[hf_mask, :] ** 2)
    total_energy = np.sum(S_mag ** 2) + 1e-12
    hf_energy_ratio = hf_energy / total_energy

    # Per-mel-band temporal variance (averaged)
    mel_band_temporal_std = np.mean(np.std(mel_db, axis=1))

    # Centroid-to-rolloff ratio
    sc_mean = float(np.mean(spec_centroid))
    sr_mean = float(np.mean(spec_rolloff))
    centroid_rolloff_ratio = sc_mean / (sr_mean + 1e-9)

    stats = {
        "centroid_mean_hz":       round(sc_mean, 1),
        "centroid_std_hz":        round(float(np.std(spec_centroid)), 1),
        "rolloff_mean_hz":        round(sr_mean, 1),
        "rolloff_std_hz":         round(float(np.std(spec_rolloff)), 1),
        "bandwidth_mean_hz":      round(float(np.mean(spec_bandwidth)), 1),
        "bandwidth_std_hz":       round(float(np.std(spec_bandwidth)), 1),
        "centroid_rolloff_ratio":  round(float(centroid_rolloff_ratio), 4),
        "hf_energy_ratio":         round(float(hf_energy_ratio), 4),
        "spectral_contrast_bands": [round(float(c), 2) for c in contrast_band_means],
        "mel_band_temporal_std":   round(float(mel_band_temporal_std), 2),
    }

    return dict(
        y=y, sr=sr, duration=duration,
        frame_times=frame_times,
        S_db=S_db, mel_db=mel_db,
        spec_centroid=spec_centroid,
        spec_rolloff=spec_rolloff,
        spec_bandwidth=spec_bandwidth,
        hop_length=hop_length,
        stats=stats,
    )


def plot_spectral(data: dict, audio_path: str, output_path: str = None):
    filename = audio_path.split("/")[-1]
    sr  = data["sr"]
    hop = data["hop_length"]
    dur = data["duration"]
    ft  = data["frame_times"]
    stats = data["stats"]

    sc  = data["spec_centroid"]
    sr_f = data["spec_rolloff"]
    sb  = data["spec_bandwidth"]
    ft, sc, sr_f, sb = trim_to_shortest(ft, sc, sr_f, sb)

    fig, axes = make_figure(4, height_per_panel=3.5,
                            title=f"Spectral & Brightness Analysis — {filename}")
    a0, a1, a2, a3 = axes

    # ── Panel 1: STFT ────────────────────────────────────────────────────────
    img = librosa.display.specshow(data["S_db"], sr=sr, hop_length=hop,
                                   x_axis="time", y_axis="log", cmap="magma", ax=a0)
    fig.colorbar(img, ax=a0, format="%+2.0f dB", pad=0.01).ax.tick_params(
        labelsize=7, colors="#aaa")
    a0.set_title("STFT Spectrogram (log frequency, dB)", fontsize=9)
    a0.set_ylabel("Freq (Hz)", fontsize=8)

    # ── Panel 2: Mel ─────────────────────────────────────────────────────────
    img = librosa.display.specshow(data["mel_db"], sr=sr, hop_length=hop,
                                   x_axis="time", y_axis="mel", cmap="inferno", ax=a1)
    fig.colorbar(img, ax=a1, format="%+2.0f dB", pad=0.01).ax.tick_params(
        labelsize=7, colors="#aaa")
    a1.set_title("Mel Spectrogram (perceptual frequency scale)", fontsize=9)
    a1.set_ylabel("Mel", fontsize=8)

    # ── Panel 3: Centroid / Rolloff / Bandwidth ──────────────────────────────
    a2.plot(ft, sc,  color="#4fc3f7", lw=0.8, label="Centroid")
    a2.plot(ft, sr_f, color="#f06292", lw=0.8, label="Rolloff (85%)")
    a2.fill_between(ft, sc - sb, sc + sb, color="#4fc3f7", alpha=0.15, label="±Bandwidth")
    a2.set_ylabel("Freq (Hz)", fontsize=8)
    a2.set_title("Spectral Centroid / Rolloff / Bandwidth — brightness & spread", fontsize=9)
    a2.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    time_axis(a2, dur)

    # ── Panel 4: Stats text ──────────────────────────────────────────────────
    a3.axis("off")
    contrast_str = ", ".join(str(c) for c in stats["spectral_contrast_bands"])
    lines = [
        ("Centroid Mean",         f"{stats['centroid_mean_hz']} Hz"),
        ("Centroid Std",          f"{stats['centroid_std_hz']} Hz   ← low = static brightness (AI trait)"),
        ("Rolloff Mean",          f"{stats['rolloff_mean_hz']} Hz"),
        ("Rolloff Std",           f"{stats['rolloff_std_hz']} Hz"),
        ("Bandwidth Mean",        f"{stats['bandwidth_mean_hz']} Hz"),
        ("Bandwidth Std",         f"{stats['bandwidth_std_hz']} Hz"),
        ("Centroid/Rolloff Ratio",f"{stats['centroid_rolloff_ratio']}"),
        ("HF Energy Ratio (>8k)", f"{stats['hf_energy_ratio']}   ← abrupt cutoff = codec artifact"),
        ("Mel-Band Temporal Std",  f"{stats['mel_band_temporal_std']}   ← low = static texture (AI trait)"),
        ("Spectral Contrast",      contrast_str),
    ]
    col_x = [0.02, 0.25]
    y_pos = 0.95
    a3.text(0.5, 1.0, "Spectral Summary", ha="center", va="top",
            color="white", fontsize=9, transform=a3.transAxes, weight="bold")
    for label, value in lines:
        a3.text(col_x[0], y_pos, label, color="#aaa", fontsize=8.5,
                transform=a3.transAxes, va="top")
        a3.text(col_x[1], y_pos, value, color="white", fontsize=8.5,
                transform=a3.transAxes, va="top")
        y_pos -= 0.08

    save_or_show(fig, output_path)

    print("\n── Spectral Stats ─────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<26} {v}")
    print("────────────────────────────────────────────────────────\n")


def main():
    parser = base_argparser("Spectral & brightness analysis for AI vs. human music research.")
    args = parser.parse_args()
    print(f"[spectral] Analyzing: {args.audio_path}")
    data = analyze_spectral(args.audio_path)
    plot_spectral(data, args.audio_path, output_path=args.output)


if __name__ == "__main__":
    main()