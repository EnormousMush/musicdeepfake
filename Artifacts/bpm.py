"""
BPM estimation with optional backends:
- madmom (recommended): RNN beat activations + TempoEstimationProcessor. Install: pip install madmom
- librosa: Essentia-style histogram from beat_track, or tempo/beat_track methods.
"""
import argparse
from typing import Optional

import numpy as np

# madmom 0.16.x uses deprecated NumPy aliases (np.float, etc) which are removed in NumPy 2.0.
# Patch them back at runtime so importing madmom works without downgrading NumPy.
_NUMPY_LEGACY_ALIASES = {
    "bool": bool,
    "complex": complex,
    "float": float,
    "int": int,
    "object": object,
    "str": str,
}
for _name, _typ in _NUMPY_LEGACY_ALIASES.items():
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

try:
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor
    HAS_MADMOM = True
except ImportError:
    HAS_MADMOM = False

import librosa

try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Essentia RhythmExtractor2013 uses 44100 Hz; many beat detectors are tuned for it.
RHYTHM_SR = 44100
DEFAULT_MIN_BPM = 40
DEFAULT_MAX_BPM = 208
# madmom RNNBeatProcessor output frame rate
MADMOM_FPS = 100


def _estimate_bpm_madmom(
    audio_path: str,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> float:
    """
    BPM via madmom: RNN beat activations + TempoEstimationProcessor (comb filter).
    Often more accurate than librosa; requires: pip install madmom
    """
    rnn = RNNBeatProcessor()
    tempo_proc = TempoEstimationProcessor(
        fps=MADMOM_FPS, min_bpm=min_bpm, max_bpm=max_bpm, method="comb"
    )
    act = rnn(audio_path)
    tempi = tempo_proc(act)
    if tempi is None or len(tempi) == 0:
        return float(min_bpm + max_bpm) / 2.0
    # First row: [bpm, strength]; take dominant tempo
    bpm = float(tempi[0, 0])
    # Use same half/double folding as other methods so half-time estimates
    # (e.g. ~46.5 for a 93 BPM track) can be corrected via min/max tempo.
    return _constrain_bpm(bpm, min_bpm, max_bpm)


def _to_scalar(x) -> float:
    return float(np.atleast_1d(x).flat[0])


def _load_for_rhythm(audio_path: str):
    """Load mono at 44100 Hz for rhythm/tempo (Essentia-style)."""
    y, sr = librosa.load(audio_path, sr=RHYTHM_SR, mono=True)
    return y, sr


def _constrain_bpm(bpm: float, min_bpm: float, max_bpm: float) -> float:
    """Fold half/double tempo into [min_bpm, max_bpm] and clamp."""
    while bpm < min_bpm and bpm > 0:
        bpm *= 2
    while bpm > max_bpm:
        bpm /= 2
    return max(min_bpm, min(max_bpm, bpm))


def _uniform_prior(min_bpm: float, max_bpm: float):
    """Uniform prior over [min_bpm, max_bpm] so no BPM is favored (avoids 120 bias)."""
    if not HAS_SCIPY:
        return None
    return scipy.stats.uniform(loc=min_bpm, scale=max_bpm - min_bpm)


def _estimate_bpm_tempo(
    audio_path: str,
    prior_bpm: float = 120.0,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> float:
    """Onset-strength tempo. Uses uniform prior so audio drives result, not a fixed 120."""
    y, sr = _load_for_rhythm(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    prior = _uniform_prior(min_bpm, max_bpm)
    if prior is not None:
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, prior=prior, aggregate=np.median
        )
    else:
        tempo = librosa.beat.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=np.median
        )
    return _constrain_bpm(_to_scalar(tempo), min_bpm, max_bpm)


def _estimate_bpm_beat_track(
    audio_path: str,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> float:
    """Beat tracking tempo. Uses uniform prior so result varies with the song."""
    y, sr = _load_for_rhythm(audio_path)
    prior = _uniform_prior(min_bpm, max_bpm)
    try:
        if prior is not None:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, prior=prior)
        else:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except TypeError:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return _constrain_bpm(_to_scalar(tempo), min_bpm, max_bpm)


# Essentia RhythmExtractor2013: "MAGIC NUMBER" period tolerance in BPM
_PERIOD_TOLERANCE_BPM = 5.0


def _estimate_bpm_histogram(
    audio_path: str,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> float:
    """
    BPM from inter-beat intervals using Essentia RhythmExtractor2013 logic:
    - intervals -> BPM per interval
    - divide BPMs by 2, bincount (histogram), argmax * 2 = closestBpm
    - keep estimates within periodTolerance (5 BPM) of closestBpm
    - return mean(estimates) or closestBpm if none in range
    """
    y, sr = _load_for_rhythm(audio_path)
    prior = _uniform_prior(min_bpm, max_bpm)
    try:
        if prior is not None:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, prior=prior)
        else:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    except TypeError:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 2:
        return _constrain_bpm(_to_scalar(tempo), min_bpm, max_bpm)
    # Beat intervals in seconds -> BPM per interval (as in Essentia)
    bpm_intervals_sec = np.diff(beat_times)
    bpm_intervals_sec = bpm_intervals_sec[bpm_intervals_sec > 0.15]
    if len(bpm_intervals_sec) == 0:
        return _constrain_bpm(_to_scalar(tempo), min_bpm, max_bpm)
    bpm_estimate_list = 60.0 / bpm_intervals_sec
    # Essentia: divide by 2, bincount, argmax * 2 -> closestBpm (bin index = half-BPM)
    halved = bpm_estimate_list / 2.0
    bin_edges = np.arange(0, int(max_bpm / 2) + 2, 1.0)
    hist, _ = np.histogram(halved, bins=bin_edges)
    peak_bin = int(np.argmax(hist))
    closest_bpm = float(peak_bin * 2)  # Essentia: argmax(countedBins) * 2
    # Keep only estimates within periodTolerance of closestBpm
    in_tolerance = np.abs(bpm_estimate_list - closest_bpm) < _PERIOD_TOLERANCE_BPM
    estimates = bpm_estimate_list[in_tolerance]
    if len(estimates) < 1:
        bpm = closest_bpm
    else:
        bpm = float(np.mean(estimates))
    return _constrain_bpm(bpm, min_bpm, max_bpm)


def _default_method() -> str:
    """Use madmom if available, else librosa histogram."""
    return "madmom" if HAS_MADMOM else "histogram"


def estimate_bpm(
    audio_path: str,
    prior_bpm: Optional[float] = 120.0,
    method: Optional[str] = None,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
) -> float:
    """
    Estimate the tempo (BPM) of an audio file.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (wav, mp3, etc.)
    prior_bpm : float, optional
        Expected BPM hint for tempo method (default 120).
    method : str, optional
        "madmom" (recommended, if installed), "tempo", "beat_track", "histogram",
        or "combined". Default: madmom if available, else histogram.
    min_bpm, max_bpm : float
        Tempo range (default 40–208).

    Returns
    -------
    float
        Estimated tempo in beats per minute.
    """
    if method is None:
        method = _default_method()
    if method == "madmom":
        if not HAS_MADMOM:
            raise RuntimeError("madmom not installed. Run: pip install madmom")
        return _estimate_bpm_madmom(audio_path, min_bpm, max_bpm)
    if method == "beat_track":
        return _estimate_bpm_beat_track(audio_path, min_bpm, max_bpm)
    if method == "histogram":
        return _estimate_bpm_histogram(audio_path, min_bpm, max_bpm)
    if method == "combined":
        est = [_estimate_bpm_tempo(audio_path, prior_bpm or 120.0, min_bpm, max_bpm),
               _estimate_bpm_beat_track(audio_path, min_bpm, max_bpm),
               _estimate_bpm_histogram(audio_path, min_bpm, max_bpm)]
        if HAS_MADMOM:
            est.append(_estimate_bpm_madmom(audio_path, min_bpm, max_bpm))
        return float(np.median(est))
    return _estimate_bpm_tempo(audio_path, prior_bpm or 120.0, min_bpm, max_bpm)


def _candidates(bpm: float, min_bpm: float = 40, max_bpm: float = 240) -> list[float]:
    """Return BPM and common half/double variants in plausible range."""
    out = {bpm, bpm / 2, bpm * 2}
    return sorted(v for v in out if min_bpm <= v <= max_bpm)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate BPM. Uses madmom if installed, else librosa."
    )
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument(
        "--prior",
        type=float,
        default=120,
        help="Expected BPM hint for librosa tempo method (default 120).",
    )
    methods = ["madmom", "tempo", "beat_track", "histogram", "combined"]
    if not HAS_MADMOM:
        methods = [m for m in methods if m != "madmom"]
    parser.add_argument(
        "--method",
        choices=methods,
        default=_default_method(),
        help="Backend: madmom (if installed), tempo, beat_track, histogram, or combined.",
    )
    parser.add_argument(
        "--min-tempo",
        type=float,
        default=DEFAULT_MIN_BPM,
        dest="min_bpm",
        help="Minimum BPM (default 40, per Essentia).",
    )
    parser.add_argument(
        "--max-tempo",
        type=float,
        default=DEFAULT_MAX_BPM,
        dest="max_bpm",
        help="Maximum BPM (default 208, per Essentia).",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="Show plausible BPMs from tempo, beat_track, and histogram.",
    )
    args = parser.parse_args()

    if args.candidates:
        all_bpm = []
        t = _estimate_bpm_tempo(args.audio_path, args.prior, args.min_bpm, args.max_bpm)
        b = _estimate_bpm_beat_track(args.audio_path, args.min_bpm, args.max_bpm)
        h = _estimate_bpm_histogram(args.audio_path, args.min_bpm, args.max_bpm)
        all_bpm = (
            _candidates(t, args.min_bpm, args.max_bpm)
            + _candidates(b, args.min_bpm, args.max_bpm)
            + _candidates(h, args.min_bpm, args.max_bpm)
        )
        if HAS_MADMOM:
            try:
                m = _estimate_bpm_madmom(args.audio_path, args.min_bpm, args.max_bpm)
                all_bpm += _candidates(m, args.min_bpm, args.max_bpm)
            except Exception:
                pass
        seen = set()
        for x in sorted(set(round(v, 1) for v in all_bpm)):
            if x not in seen:
                seen.add(x)
                print(f"{x:.1f}")
        return

    bpm = estimate_bpm(
        args.audio_path,
        prior_bpm=args.prior,
        method=args.method,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
    )
    print(f"{bpm:.2f}")


if __name__ == "__main__":
    main()
