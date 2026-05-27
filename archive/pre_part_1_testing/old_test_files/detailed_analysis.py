"""
Comprehensive Waveform & Audio Analysis

Generates a multi-panel analysis figure designed to surface artifacts that may
distinguish AI-generated music from human-recorded music.

Panels produced:
  1.  Waveform + RMS envelope + onset markers
  2.  Short-Time Fourier Transform (STFT) spectrogram
  3.  Mel-frequency spectrogram
  4.  Chromagram (harmonic/pitch-class energy)
  5.  MFCCs (timbral texture, 20 coefficients)
  6.  Spectral centroid + rolloff + bandwidth over time
  7.  Zero-crossing rate + spectral flatness (noisiness)
  8.  Harmonic vs. percussive energy over time
  9.  Onset strength envelope + beat markers
  10. Dynamic range: rolling peak-to-RMS crest factor
  11. Summary statistics panel (printed + embedded)

Usage:
    python waveform.py audio.mp3
    python waveform.py audio.mp3 --output analysis.png
"""
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")   # headless-safe; swap to "TkAgg" if you need a live window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_time(x, _):
    return f"{int(x // 60)}:{int(x % 60):02d}"


def _time_axis(ax, duration):
    ax.set_xlim(0, duration)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_time))
    ax.grid(axis="x", linestyle="--", alpha=0.25)


# ── core analysis ─────────────────────────────────────────────────────────────

def analyze(audio_path: str, sr_target: int = 22050, hop_length: int = 512):
    """Load audio and compute all features. Returns a dict of arrays."""
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    times_samples = np.linspace(0, duration, num=len(y))

    # HPSS
    y_harm, y_perc = librosa.effects.hpss(y, margin=4)

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
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop_length)

    # Frame times for feature arrays
    n_frames = 1 + len(y) // hop_length
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # RMS
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

    # Spectral features
    spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spec_rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    spec_flatness  = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    zcr            = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)

    # Chroma (CQT)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=24, hop_length=hop_length)

    # STFT spectrogram
    D = librosa.stft(y, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Harmonic / percussive RMS
    rms_harm = librosa.feature.rms(y=y_harm, frame_length=2048, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=hop_length)[0]

    # Crest factor (rolling peak / RMS) — high = more dynamic, low = compressed/uniform
    # Use same frames as rms
    peak_env = np.array([
        np.max(np.abs(y[max(0, i * hop_length - 1024): i * hop_length + 1024]))
        for i in range(len(rms))
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 1e-6, peak_env / rms, 0.0)

    # Summary stats
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))
    stats = {
        "tempo_bpm":        round(tempo, 1),
        "duration_s":       round(duration, 2),
        "num_onsets":       len(onset_times),
        "num_beats":        len(beat_times),
        "rms_mean_db":      round(float(np.mean(rms_db)), 1),
        "rms_std_db":       round(float(np.std(rms_db)), 2),
        "dynamic_range_db": round(float(np.max(rms_db) - np.min(rms_db[rms_db > -60])), 1),
        "crest_mean":       round(float(np.mean(crest[crest > 0])), 2),
        "crest_std":        round(float(np.std(crest[crest > 0])), 2),
        "zcr_mean":         round(float(np.mean(zcr)), 4),
        "spec_flat_mean":   round(float(np.mean(spec_flatness)), 4),
        "centroid_mean_hz": round(float(np.mean(spec_centroid)), 1),
        "harm_perc_ratio":  round(float(np.mean(rms_harm) / (np.mean(rms_perc) + 1e-9)), 3),
    }

    return dict(
        y=y, sr=sr, duration=duration,
        times_samples=times_samples,
        frame_times=frame_times,
        beat_times=beat_times,
        onset_times=onset_times,
        onset_env=onset_env,
        rms=rms, rms_harm=rms_harm, rms_perc=rms_perc,
        crest=crest,
        spec_centroid=spec_centroid,
        spec_rolloff=spec_rolloff,
        spec_bandwidth=spec_bandwidth,
        spec_flatness=spec_flatness,
        zcr=zcr,
        mfcc=mfcc,
        chroma=chroma,
        S_db=S_db,
        mel_db=mel_db,
        hop_length=hop_length,
        stats=stats,
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_analysis(data: dict, audio_path: str, output_path: str = None):
    filename = audio_path.split("/")[-1]
    sr       = data["sr"]
    hop      = data["hop_length"]
    dur      = data["duration"]
    ft       = data["frame_times"]
    stats    = data["stats"]

    # Trim frame-level arrays to same length (STFT may differ by 1)
    def _trim(*arrays):
        n = min(len(a) for a in arrays)
        return tuple(a[:n] for a in arrays)

    ft, rms, rms_harm, rms_perc, crest, sc, sr_f, sb, sf, zcr = _trim(
        ft, data["rms"], data["rms_harm"], data["rms_perc"], data["crest"],
        data["spec_centroid"], data["spec_rolloff"], data["spec_bandwidth"],
        data["spec_flatness"], data["zcr"],
    )

    fig = plt.figure(figsize=(18, 34), facecolor="#0e1117")
    fig.suptitle(f"Comprehensive Audio Analysis\n{filename}",
                 fontsize=15, color="white", y=0.995, weight="bold")

    gs = gridspec.GridSpec(11, 1, figure=fig, hspace=0.55,
                           top=0.985, bottom=0.03, left=0.07, right=0.97)

    ax = [fig.add_subplot(gs[i]) for i in range(11)]
    for a in ax:
        a.set_facecolor("#1a1d27")
        for spine in a.spines.values():
            spine.set_edgecolor("#444")
        a.tick_params(colors="#aaa", labelsize=8)
        a.xaxis.label.set_color("#aaa")
        a.yaxis.label.set_color("#aaa")
        a.title.set_color("white")

    # ── 1. Waveform + RMS + onsets ────────────────────────────────────────────
    a = ax[0]
    a.plot(data["times_samples"], data["y"], color="#4a90d9", lw=0.3, alpha=0.7)
    rms_scaled = rms / (rms.max() + 1e-9) * np.abs(data["y"]).max()
    a.fill_between(ft,  rms_scaled, -rms_scaled,
                   color="#e8734a", alpha=0.35, label="RMS envelope")
    a.vlines(data["onset_times"], -1, 1, color="#f5c842", lw=0.6, alpha=0.55,
             label=f"Onsets ({len(data['onset_times'])})")
    a.vlines(data["beat_times"], -1, 1, color="#7aff7a", lw=0.8, alpha=0.4,
             label=f"Beats ({len(data['beat_times'])})")
    a.set_ylim(-1.1, 1.1)
    a.set_ylabel("Amplitude", fontsize=8)
    a.set_title("1 · Waveform  +  RMS Envelope  +  Onsets  +  Beats", fontsize=9)
    a.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    _time_axis(a, dur)

    # ── 2. STFT Spectrogram ───────────────────────────────────────────────────
    a = ax[1]
    img = librosa.display.specshow(
        data["S_db"], sr=sr, hop_length=hop, x_axis="time", y_axis="log",
        cmap="magma", ax=a
    )
    fig.colorbar(img, ax=a, format="%+2.0f dB", pad=0.01).ax.tick_params(labelsize=7, colors="#aaa")
    a.set_title("2 · STFT Spectrogram (log frequency, dB)", fontsize=9)
    a.set_ylabel("Freq (Hz)", fontsize=8)

    # ── 3. Mel Spectrogram ────────────────────────────────────────────────────
    a = ax[2]
    img = librosa.display.specshow(
        data["mel_db"], sr=sr, hop_length=hop, x_axis="time", y_axis="mel",
        cmap="inferno", ax=a
    )
    fig.colorbar(img, ax=a, format="%+2.0f dB", pad=0.01).ax.tick_params(labelsize=7, colors="#aaa")
    a.set_title("3 · Mel Spectrogram (perceptual frequency scale)", fontsize=9)
    a.set_ylabel("Mel", fontsize=8)

    # ── 4. Chromagram ─────────────────────────────────────────────────────────
    a = ax[3]
    img = librosa.display.specshow(
        data["chroma"], sr=sr, hop_length=hop, x_axis="time", y_axis="chroma",
        cmap="coolwarm", ax=a
    )
    fig.colorbar(img, ax=a, pad=0.01).ax.tick_params(labelsize=7, colors="#aaa")
    a.set_title("4 · Chromagram — pitch-class energy (harmonic component)", fontsize=9)
    a.set_ylabel("Pitch", fontsize=8)

    # ── 5. MFCCs ──────────────────────────────────────────────────────────────
    a = ax[4]
    img = librosa.display.specshow(
        data["mfcc"], sr=sr, hop_length=hop, x_axis="time",
        cmap="RdBu_r", ax=a
    )
    fig.colorbar(img, ax=a, pad=0.01).ax.tick_params(labelsize=7, colors="#aaa")
    a.set_title("5 · MFCCs — timbral texture (20 coefficients)", fontsize=9)
    a.set_ylabel("MFCC #", fontsize=8)

    # ── 6. Spectral centroid / rolloff / bandwidth ────────────────────────────
    a = ax[5]
    a.plot(ft, sc,   color="#4fc3f7", lw=0.8, label="Centroid")
    a.plot(ft, sr_f, color="#f06292", lw=0.8, label="Rolloff (85%)")
    a.fill_between(ft, sc - sb, sc + sb, color="#4fc3f7", alpha=0.15, label="±Bandwidth")
    a.set_ylabel("Freq (Hz)", fontsize=8)
    a.set_title("6 · Spectral Centroid / Rolloff / Bandwidth — brightness & spread", fontsize=9)
    a.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    _time_axis(a, dur)

    # ── 7. ZCR + Spectral flatness ────────────────────────────────────────────
    a = ax[6]
    ax2 = a.twinx()
    a.plot(ft, zcr, color="#aed581", lw=0.8, label="ZCR")
    ax2.plot(ft, sf, color="#ff8a65", lw=0.8, alpha=0.8, label="Flatness")
    a.set_ylabel("ZCR", fontsize=8, color="#aed581")
    ax2.set_ylabel("Flatness", fontsize=8, color="#ff8a65")
    ax2.tick_params(colors="#ff8a65", labelsize=8)
    a.set_title("7 · Zero-Crossing Rate  +  Spectral Flatness — noise vs. tonality", fontsize=9)
    lines1, labels1 = a.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    a.legend(lines1 + lines2, labels1 + labels2,
             loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    _time_axis(a, dur)

    # ── 8. Harmonic vs. percussive energy ─────────────────────────────────────
    a = ax[7]
    a.fill_between(ft, rms_harm, color="#ce93d8", alpha=0.7, label="Harmonic RMS")
    a.fill_between(ft, rms_perc, color="#80cbc4", alpha=0.5, label="Percussive RMS")
    a.set_ylabel("RMS", fontsize=8)
    a.set_title("8 · Harmonic vs. Percussive Energy (HPSS)", fontsize=9)
    a.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    _time_axis(a, dur)

    # ── 9. Onset strength + beat markers ─────────────────────────────────────
    a = ax[8]
    oe_times = librosa.frames_to_time(np.arange(len(data["onset_env"])), sr=sr, hop_length=hop)
    a.plot(oe_times, data["onset_env"], color="#ffcc02", lw=0.8, label="Onset strength")
    a.vlines(data["beat_times"], 0, data["onset_env"].max(),
             color="#7aff7a", lw=0.9, alpha=0.5, label="Beats")
    a.set_ylabel("Strength", fontsize=8)
    a.set_title("9 · Onset Strength Envelope + Beat Grid — rhythmic regularity", fontsize=9)
    a.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    _time_axis(a, dur)

    # ── 10. Crest factor (dynamic range indicator) ────────────────────────────
    a = ax[9]
    a.plot(ft, crest, color="#ef9a9a", lw=0.8, label="Crest factor (peak/RMS)")
    a.axhline(np.mean(crest[crest > 0]), color="#ef9a9a", lw=1.2,
              linestyle="--", alpha=0.6, label=f"Mean={stats['crest_mean']:.2f}")
    a.set_ylabel("Crest factor", fontsize=8)
    a.set_title(
        "10 · Crest Factor — dynamic compression  "
        "(low=brickwall-limited/AI-uniform, high=dynamic/natural)", fontsize=9
    )
    a.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white")
    _time_axis(a, dur)

    # ── 11. Summary statistics ────────────────────────────────────────────────
    a = ax[10]
    a.axis("off")
    lines = [
        ("Tempo",            f"{stats['tempo_bpm']} BPM"),
        ("Duration",         f"{stats['duration_s']} s"),
        ("Onsets",           str(stats["num_onsets"])),
        ("Beats",            str(stats["num_beats"])),
        ("RMS mean",         f"{stats['rms_mean_db']} dB"),
        ("RMS std",          f"{stats['rms_std_db']} dB  ← low = uniform loudness (AI trait)"),
        ("Dynamic range",    f"{stats['dynamic_range_db']} dB"),
        ("Crest factor μ",   f"{stats['crest_mean']}  ← low = heavily compressed"),
        ("Crest factor σ",   f"{stats['crest_std']}"),
        ("ZCR mean",         str(stats["zcr_mean"])),
        ("Spectral flatness",f"{stats['spec_flat_mean']}  ← high = noise-like, low = tonal"),
        ("Centroid mean",    f"{stats['centroid_mean_hz']} Hz"),
        ("Harm/Perc ratio",  f"{stats['harm_perc_ratio']}"),
    ]
    col_x = [0.02, 0.22]
    y_pos = 0.96
    a.text(0.5, 1.0, "11 · Summary Statistics", ha="center", va="top",
           color="white", fontsize=9, transform=a.transAxes, weight="bold")
    for label, value in lines:
        a.text(col_x[0], y_pos, label, color="#aaa",  fontsize=8.5,
               transform=a.transAxes, va="top")
        a.text(col_x[1], y_pos, value, color="white", fontsize=8.5,
               transform=a.transAxes, va="top")
        y_pos -= 0.07

    # ── save / show ───────────────────────────────────────────────────────────
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved to: {output_path}")
    else:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close()

    # Also print stats to console
    print("\n── Summary Statistics ──────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print("────────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive waveform & spectral analysis for AI vs. human music research."
    )
    parser.add_argument("audio_path", help="Path to audio file (mp3, wav, flac, etc.)")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save figure to file (e.g. analysis.png). If omitted, opens a window."
    )
    args = parser.parse_args()

    print(f"Analyzing: {args.audio_path}")
    data = analyze(args.audio_path)
    plot_analysis(data, args.audio_path, output_path=args.output)


if __name__ == "__main__":
    main()
