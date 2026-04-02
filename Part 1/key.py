"""
Key estimation for audio files using librosa's chroma features + Krumhansl-Schmuckler profiles.
Approach adapted from Tonal_Fragment (Philip Kirlin) with HPSS harmonic separation.
"""
import argparse
import numpy as np
import librosa

# Krumhansl-Schmuckler profiles (as used in Tonal_Fragment reference)
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_key(audio_path: str):
    """
    Returns (best_key, best_corr, alt_key, alt_corr).
    alt_key is set if a second key has correlation within 90% of the best.
    """
    y, sr = librosa.load(audio_path, sr=11025, mono=True)

    # Isolate harmonic component — removes drums/transients that pollute chroma
    y_harmonic, _ = librosa.effects.hpss(y, margin=4)

    # bins_per_octave=24 gives higher chroma resolution (per Tonal_Fragment)
    chromograph = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=24)

    # Sum energy per pitch class over time (per Tonal_Fragment)
    chroma_vals = [float(np.sum(chromograph[i])) for i in range(12)]
    keyfreqs = {_NOTE_NAMES[i]: chroma_vals[i] for i in range(12)}

    # Compute correlation of each major/minor key against the chroma profile
    key_dict = {}
    for i in range(12):
        key_test = [keyfreqs[_NOTE_NAMES[(i + m) % 12]] for m in range(12)]
        maj_corr = round(float(np.corrcoef(_MAJOR_PROFILE, key_test)[1, 0]), 3)
        min_corr = round(float(np.corrcoef(_MINOR_PROFILE, key_test)[1, 0]), 3)
        key_dict[f"{_NOTE_NAMES[i]} major"] = maj_corr
        key_dict[f"{_NOTE_NAMES[i]} minor"] = min_corr

    best_key = max(key_dict, key=key_dict.get)
    best_corr = key_dict[best_key]

    # Report alternative key if its correlation is within 90% of the best
    alt_key, alt_corr = None, None
    for key, corr in key_dict.items():
        if corr > best_corr * 0.9 and corr != best_corr:
            alt_key = key
            alt_corr = corr

    return best_key, best_corr, alt_key, alt_corr


def main():
    parser = argparse.ArgumentParser(description="Estimate the musical key of an audio file.")
    parser.add_argument("audio_path", help="Path to the audio file (mp3, wav, etc.)")
    args = parser.parse_args()

    best_key, best_corr, alt_key, alt_corr = estimate_key(args.audio_path)
    print(f"likely key: {best_key}, correlation: {best_corr}")
    if alt_key:
        print(f"also possible: {alt_key}, correlation: {alt_corr}")


if __name__ == "__main__":
    main()
