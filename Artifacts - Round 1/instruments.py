"""
Instrument detection using CLAP (Contrastive Language-Audio Pretraining).
Zero-shot: scores each instrument label against the audio without retraining.
Processes the song in overlapping chunks and aggregates scores.

Requires: pip install transformers torch torchaudio
Model (~900MB) is downloaded automatically on first run.
"""
import argparse
import numpy as np
import librosa

_SAMPLE_RATE = 48000  # CLAP expects 48kHz
_CHUNK_DURATION = 10  # seconds per chunk
_HOP_DURATION = 5     # seconds between chunk starts (50% overlap)

_DEFAULT_INSTRUMENTS = [
    "acoustic guitar", "electric guitar", "bass guitar",
    "piano", "keyboard", "organ",
    "drums", "percussion",
    "violin", "cello", "string section",
    "trumpet", "trombone", "brass",
    "saxophone", "clarinet", "flute",
    "vocals", "singing",
    "synthesizer",
]


def detect_instruments(
    audio_path: str,
    instruments: list = None,
    top_k: int = 5,
    threshold: float = 0.15,
) -> list:
    """
    Returns list of (instrument, score) tuples sorted by confidence.
    Splits the song into overlapping chunks, scores each, takes the max
    score per instrument across all chunks.
    """
    from transformers import pipeline

    labels = instruments or _DEFAULT_INSTRUMENTS

    # Load audio at 48kHz mono
    y, sr = librosa.load(audio_path, sr=_SAMPLE_RATE, mono=True)

    chunk_size = int(_CHUNK_DURATION * _SAMPLE_RATE)
    hop_size = int(_HOP_DURATION * _SAMPLE_RATE)

    # Split into overlapping chunks
    chunks = []
    start = 0
    while start < len(y):
        chunk = y[start: start + chunk_size]
        if len(chunk) < _SAMPLE_RATE:  # skip chunks shorter than 1 second
            break
        chunks.append(chunk)
        start += hop_size

    if not chunks:
        chunks = [y]

    classifier = pipeline(
        task="zero-shot-audio-classification",
        model="laion/clap-htsat-unfused",
    )

    # Accumulate max score per label across all chunks
    max_scores = {label: 0.0 for label in labels}
    for chunk in chunks:
        results = classifier(chunk, candidate_labels=labels,
                             sampling_rate=_SAMPLE_RATE)
        for item in results:
            lbl = item["label"]
            score = item["score"]
            if score > max_scores.get(lbl, 0.0):
                max_scores[lbl] = score

    # Filter by threshold and sort
    detected = [(lbl, score) for lbl, score in max_scores.items() if score >= threshold]
    detected.sort(key=lambda x: x[1], reverse=True)
    return detected[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Detect instruments in an audio file using CLAP zero-shot classification."
    )
    parser.add_argument("audio_path", help="Path to the audio file (mp3, wav, etc.)")
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top instruments to show (default: 5)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Minimum confidence score 0-1 (default: 0.15)"
    )
    parser.add_argument(
        "--instruments", nargs="+",
        help="Custom list of instruments to check (e.g. --instruments piano guitar violin)"
    )
    args = parser.parse_args()

    detected = detect_instruments(
        args.audio_path,
        instruments=args.instruments,
        top_k=args.top,
        threshold=args.threshold,
    )

    if not detected:
        print("No instruments detected above threshold.")
    else:
        print("Detected instruments:")
        for label, score in detected:
            print(f"  {label:<30} {score:.3f}")


if __name__ == "__main__":
    main()
