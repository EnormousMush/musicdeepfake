"""
【2026-08-20 冻结】特征提取已迁往 part2_analysis/features/families/quantize.py(公式原样,对拍逐位一致;load/HPSS/beat/onset 改共享 FeatureContext)。

Degree of Quantization Analyzer

Measures how "robotically" timed an audio file is by:
1. Detecting onsets (note hits)
2. Building a local beat grid (adaptive to tempo changes)
3. Computing deviation of each onset from the nearest grid subdivision
4. Measuring swing ratio (straight vs. swung 8th notes)

Output:
  - Quantization score (0-100): 100 = perfectly on-grid, 0 = random timing
  - Mean/std/max deviation from grid (in milliseconds)
  - Swing ratio: 50% = straight, 67% = triplet swing
"""
import argparse
import numpy as np
import librosa


def _build_local_grid(beat_times: np.ndarray, subdivisions: int) -> np.ndarray:
    """
    Build a subdivision grid that adapts to local tempo changes.
    Each beat interval is subdivided into `subdivisions` equal parts.
    """
    grid = []
    for i in range(len(beat_times) - 1):
        beat_dur = beat_times[i + 1] - beat_times[i]
        for s in range(subdivisions):
            grid.append(beat_times[i] + s * beat_dur / subdivisions)
    # Extend one more beat using last known beat duration
    if len(beat_times) >= 2:
        last_dur = beat_times[-1] - beat_times[-2]
        for s in range(subdivisions):
            grid.append(beat_times[-1] + s * last_dur / subdivisions)
    return np.array(grid)


def _swing_ratio(beat_times: np.ndarray, onset_times: np.ndarray) -> float:
    """
    Measure swing as the average position of off-beat onsets relative to the beat.
      50%  = perfectly straight
      66.7% = triplet (hard) swing
    """
    ratios = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_dur = beat_times[i + 1] - beat_times[i]

        # Find onsets in the off-beat window (30%-80% into the beat)
        off_beat_onsets = onset_times[
            (onset_times > beat_start + beat_dur * 0.3) &
            (onset_times < beat_start + beat_dur * 0.8)
        ]
        if len(off_beat_onsets) > 0:
            off_beat = off_beat_onsets[0]
            ratios.append((off_beat - beat_start) / beat_dur)

    return float(np.mean(ratios)) if ratios else 0.5


def analyze_quantization(audio_path: str, subdivision: int = 16) -> dict:
    """
    Analyze the degree of quantization in an audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file (mp3, wav, etc.)
    subdivision : int
        Grid resolution: 8 = 8th notes, 16 = 16th notes (default), 32 = 32nd notes

    Returns
    -------
    dict with keys:
        tempo, quantization_score, mean_dev_ms, std_dev_ms, max_dev_ms,
        swing_pct, num_onsets, subdivision
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Percussive component gives cleaner onset detection
    _, y_perc = librosa.effects.hpss(y, margin=4)

    # Detect tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 2:
        return {"error": "Not enough beats detected to analyze quantization."}

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(
        y=y_perc, sr=sr, units="frames",
        pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.06, wait=10,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Only analyze onsets within the detected beat range
    onset_times = onset_times[
        (onset_times >= beat_times[0]) & (onset_times <= beat_times[-1])
    ]

    if len(onset_times) == 0:
        return {"error": "No onsets detected."}

    # Build local adaptive grid
    # beat_track gives quarter-note beats, so subdivisions_per_beat = subdivision / 4
    subs_per_beat = max(1, subdivision // 4)
    grid = _build_local_grid(beat_times, subs_per_beat)

    # Compute deviation of each onset from nearest grid point
    deviations_sec = np.array([
        onset - grid[np.argmin(np.abs(grid - onset))]
        for onset in onset_times
    ])
    abs_devs = np.abs(deviations_sec)

    # Max possible deviation = half a grid cell
    beat_dur_avg = float(np.mean(np.diff(beat_times)))
    grid_cell = beat_dur_avg / subs_per_beat
    max_possible_dev = grid_cell / 2.0

    mean_dev = float(np.mean(abs_devs))
    std_dev  = float(np.std(deviations_sec))
    max_dev  = float(np.max(abs_devs))

    quant_score = max(0.0, 100.0 * (1.0 - mean_dev / max_possible_dev))
    swing_pct   = _swing_ratio(beat_times, onset_times) * 100.0

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


def main():
    parser = argparse.ArgumentParser(
        description="Measure the degree of quantization in an audio file."
    )
    parser.add_argument("audio_path", help="Path to audio file (mp3, wav, etc.)")
    parser.add_argument(
        "--subdivision", type=int, default=16, choices=[4, 8, 16, 32],
        help="Grid resolution: 4=quarter, 8=8th, 16=16th (default), 32=32nd notes"
    )
    args = parser.parse_args()

    result = analyze_quantization(args.audio_path, subdivision=args.subdivision)

    if "error" in result:
        print("Error:", result["error"])
        return

    print(f"Tempo:               {result['tempo']} BPM")
    print(f"Quantization score:  {result['quantization_score']} / 100")
    print(f"Mean deviation:      {result['mean_dev_ms']} ms")
    print(f"Std deviation:       {result['std_dev_ms']} ms")
    print(f"Max deviation:       {result['max_dev_ms']} ms")
    print(f"Swing:               {result['swing_pct']}%  (50% = straight, 67% = triplet swing)")
    print(f"Onsets analyzed:     {result['num_onsets']}  (at 1/{result['subdivision']} note grid)")


if __name__ == "__main__":
    main()
