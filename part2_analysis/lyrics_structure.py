"""
【2026-08-20 注记】本模块从未被任何提取管线调用(零消费方);歌词结构属加人声后的 late plan,保留待启用。

Song Structure Detector
=======================
Detects the repeating structural sections of a song (Intro, Verse, Chorus,
Bridge, Outro, etc.) from the audio alone — no lyrics file needed.

Method
------
1. Extract MFCCs + chroma features (timbre + harmony)
2. Build a self-similarity / recurrence matrix
3. Detect segment boundaries via librosa's spectral-clustering segmenter
4. Compare each segment against every other using cosine similarity
5. Cluster similar segments → assign letters (A, B, C …)
6. Heuristically label: most-repeated section = Chorus, first unique = Intro,
   last unique = Outro, rare middle sections = Bridge, rest = Verse

Output
------
- Console: timestamped section list with labels
- PNG:      timeline bar + self-similarity matrix

Limitations
-----------
- Works best on structured pop/rock/EDM with clear repeated sections
- Accuracy degrades on through-composed or highly varied music
- Labels (Verse/Chorus/Bridge) are heuristic — order & repetition count drive them
- Does NOT read or transcribe actual lyric text

Usage
-----
    python lyrics_structure.py song.mp3
    python lyrics_structure.py song.mp3 --output structure.png --n-sections 8
"""

import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# ── constants ────────────────────────────────────────────────────────────────

_SR        = 11025   # half rate — sufficient for structure detection, 2x faster load
_HOP       = 2048    # larger hop → fewer frames → recurrence matrix 64x smaller
_COLORS    = [
    "#4a90d9", "#e8734a", "#7aff7a", "#f5c842", "#ce93d8",
    "#80cbc4", "#ff8a65", "#aed581", "#ef9a9a", "#4fc3f7",
]
_SIM_THRESH = 0.82   # cosine similarity to consider two segments "the same"


# ── feature extraction ────────────────────────────────────────────────────────

def _extract_features(y, sr, hop=_HOP):
    """Return stacked MFCC + chroma feature matrix, column = one frame."""
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    # chroma_stft is STFT-based (much faster than chroma_cqt which uses CQT)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)

    # Normalise each feature type independently then stack
    mfcc_n   = normalize(mfcc,   norm="l2", axis=1)
    chroma_n = normalize(chroma, norm="l2", axis=1)
    return np.vstack([mfcc_n, chroma_n])   # shape: (25, T)


# ── boundary detection ────────────────────────────────────────────────────────

def _detect_boundaries(features, sr, hop, n_sections):
    """
    Use librosa's recurrence-based segmentation to find structural boundaries.
    Returns boundary times in seconds.
    """
    # Recurrence matrix (self-similarity) for structure finding
    R = librosa.segment.recurrence_matrix(
        features, width=3, mode="affinity", sym=True
    )
    # path_enhance removed — it was computed but never used, and is O(T²)

    # Agglomerative boundary detection on the Laplacian eigenvectors
    bounds_frames = librosa.segment.agglomerative(features, k=n_sections)
    bounds_frames = np.unique(np.concatenate([[0], bounds_frames,
                                              [features.shape[1] - 1]]))
    bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop)
    return bound_times, R


# ── segment similarity & labelling ───────────────────────────────────────────

def _segment_centroids(features, bound_times, sr, hop):
    """Return mean feature vector for each segment."""
    bound_frames = librosa.time_to_frames(bound_times, sr=sr, hop_length=hop)
    centroids = []
    for i in range(len(bound_frames) - 1):
        start = bound_frames[i]
        end   = bound_frames[i + 1]
        seg   = features[:, start:end]
        centroids.append(np.mean(seg, axis=1))
    return np.array(centroids)   # shape: (n_segs, n_features)


def _assign_letters(centroids, threshold=_SIM_THRESH):
    """
    Greedily assign a letter (A, B, C…) to each segment based on cosine similarity.
    Two segments share a letter if their similarity exceeds threshold.
    """
    n = len(centroids)
    sim = cosine_similarity(centroids)
    labels = [""] * n
    letter_idx = 0

    for i in range(n):
        if labels[i]:
            continue
        letter = chr(ord("A") + letter_idx)
        labels[i] = letter
        for j in range(i + 1, n):
            if not labels[j] and sim[i, j] >= threshold:
                labels[j] = letter
        letter_idx += 1

    return labels, sim


def _heuristic_names(labels, bound_times):
    """
    Map letters → section names using position + repetition heuristics.

    Rules (applied in order of confidence):
      - Most-repeated letter → Chorus
      - First segment (if unique or short) → Intro
      - Last segment (if unique or short) → Outro
      - Rare middle segments (appear once, sandwiched) → Bridge
      - Remaining segments → Verse
    """
    from collections import Counter
    counts = Counter(labels)
    n = len(labels)
    duration = bound_times[-1] - bound_times[0]

    # Most repeated = Chorus
    chorus_letter = counts.most_common(1)[0][0]

    # Segment durations
    durations = np.diff(bound_times)

    # Intro heuristic: first segment shorter than 25% avg and appears ≤2 times
    avg_dur = np.mean(durations)
    is_intro_candidate = (
        durations[0] < avg_dur * 1.5 and counts[labels[0]] <= 2
    )
    is_outro_candidate = (
        durations[-1] < avg_dur * 1.5 and counts[labels[-1]] <= 2
    )

    intro_letter = labels[0]  if is_intro_candidate else None
    outro_letter = labels[-1] if is_outro_candidate else None

    # Bridge: appears exactly once, not intro/outro, in the latter half
    bridge_letters = set()
    for letter, cnt in counts.items():
        if cnt == 1 and letter != chorus_letter:
            # Find its position
            idx = labels.index(letter)
            position = (bound_times[idx] - bound_times[0]) / duration
            if 0.25 < position < 0.85:
                bridge_letters.add(letter)

    def name_of(letter, pos):
        if letter == chorus_letter:
            return "Chorus"
        if letter == intro_letter and pos == 0:
            return "Intro"
        if letter == outro_letter and pos == n - 1:
            return "Outro"
        if letter in bridge_letters:
            return "Bridge"
        return "Verse"

    section_names = [name_of(labels[i], i) for i in range(n)]
    return section_names


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_structure(audio_path: str, n_sections: int = 10):
    """
    Detect song structure and return a list of section dicts.

    Each dict has: start, end, letter, name, duration
    """
    y, sr = librosa.load(audio_path, sr=_SR, mono=True)
    features = _extract_features(y, sr)

    bound_times, R = _detect_boundaries(features, sr, _HOP, n_sections)
    centroids = _segment_centroids(features, bound_times, sr, _HOP)

    if len(centroids) < 2:
        return [], R, bound_times

    letters, sim = _assign_letters(centroids)
    names  = _heuristic_names(letters, bound_times)

    sections = []
    for i in range(len(letters)):
        sections.append({
            "start":    round(float(bound_times[i]),     2),
            "end":      round(float(bound_times[i + 1]), 2),
            "duration": round(float(bound_times[i + 1] - bound_times[i]), 2),
            "letter":   letters[i],
            "name":     names[i],
        })

    return sections, R, bound_times


# ── plotting ──────────────────────────────────────────────────────────────────

def _fmt(sec):
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def plot_structure(sections, R, audio_path: str, output_path: str = None):
    filename = audio_path.split("/")[-1]
    duration = sections[-1]["end"] if sections else 0

    # Assign a fixed color per letter
    letters = sorted(set(s["letter"] for s in sections))
    color_map = {l: _COLORS[i % len(_COLORS)] for i, l in enumerate(letters)}

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [1, 3]},
        facecolor="#0e1117"
    )
    fig.suptitle(f"Song Structure — {filename}", color="white", fontsize=13, weight="bold")

    # ── Panel 1: Section timeline bar ─────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a1d27")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title("Detected Section Timeline", color="white", fontsize=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: _fmt(x))
    )
    ax.tick_params(colors="#aaa", labelsize=8)

    for s in sections:
        width = s["end"] - s["start"]
        color = color_map[s["letter"]]
        rect  = mpatches.FancyBboxPatch(
            (s["start"], 0.05), width, 0.9,
            boxstyle="round,pad=0.005", linewidth=0.5,
            edgecolor="white", facecolor=color, alpha=0.85
        )
        ax.add_patch(rect)
        # Label if wide enough
        if width > duration * 0.04:
            ax.text(
                s["start"] + width / 2, 0.5,
                f"{s['name']}\n({s['letter']})",
                ha="center", va="center", fontsize=7.5, color="white", weight="bold"
            )

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[l], label=l) for l in letters
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=7, framealpha=0.3, labelcolor="white")

    # ── Panel 2: Self-similarity matrix ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#1a1d27")
    ax2.imshow(R, origin="lower", aspect="auto", cmap="magma", interpolation="nearest")

    # Overlay boundary lines on the similarity matrix
    n_frames = R.shape[0]
    if sections:
        bound_times_all = [0.0] + [s["end"] for s in sections]
        for bt in bound_times_all:
            bf = int(bt / duration * n_frames)
            ax2.axhline(bf, color="cyan", lw=0.7, alpha=0.6)
            ax2.axvline(bf, color="cyan", lw=0.7, alpha=0.6)

    ax2.set_title(
        "Self-Similarity Matrix (brighter = more similar)\n"
        "Repeated blocks along the diagonal = repeated song sections",
        color="white", fontsize=9
    )
    ax2.set_xlabel("Frame", fontsize=8, color="#aaa")
    ax2.set_ylabel("Frame", fontsize=8, color="#aaa")
    ax2.tick_params(colors="#aaa", labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect and label song structure (Intro/Verse/Chorus/Bridge/Outro)."
    )
    parser.add_argument("audio_path", help="Path to audio file (mp3, wav, etc.)")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save figure to file (e.g. structure.png)"
    )
    parser.add_argument(
        "--n-sections", type=int, default=10,
        help="Number of structural segments to detect (default: 10). "
             "Increase for longer/more complex songs."
    )
    args = parser.parse_args()

    print(f"Analyzing structure: {args.audio_path}")
    sections, R, _ = analyze_structure(args.audio_path, n_sections=args.n_sections)

    if not sections:
        print("Could not detect structure — track may be too short or unstructured.")
        return

    # Console output
    print(f"\n{'Start':>7}  {'End':>7}  {'Dur':>6}  {'§':>2}  Label")
    print("─" * 45)
    for s in sections:
        print(f"  {_fmt(s['start']):>5}  {_fmt(s['end']):>5}  "
              f"{s['duration']:>5.1f}s  {s['letter']:>2}  {s['name']}")
    print()

    plot_structure(sections, R, args.audio_path, output_path=args.output)


if __name__ == "__main__":
    main()
