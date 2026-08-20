"""
【2026-08-20 冻结】load_audio 的共享中间量职能已由 features/context.py 接管;本文件只服务旧模块的画图诊断路径。

analysis_utils.py — Shared utilities for all analysis modules.

Provides:
  - Audio loading (mono, resampled)
  - HPSS pre-split
  - Common time-axis formatting
  - Dark-theme axis styling
  - CLI argument parsing helper
"""

import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── audio loading ─────────────────────────────────────────────────────────────

def load_audio(audio_path: str, sr_target: int = 22050):
    """Load audio file, return (y, sr, duration, y_harmonic, y_percussive)."""
    y, sr = librosa.load(audio_path, sr=sr_target, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y, margin=4)
    return y, sr, duration, y_harm, y_perc


# ── formatting / axis helpers ─────────────────────────────────────────────────

def fmt_time(x, _):
    """Format seconds as M:SS for axis labels."""
    return f"{int(x // 60)}:{int(x % 60):02d}"


def time_axis(ax, duration):
    """Apply time formatting and light grid to an axis."""
    ax.set_xlim(0, duration)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_time))
    ax.grid(axis="x", linestyle="--", alpha=0.25)


def style_axes(axes, title=""):
    """Apply dark theme to a list of axes."""
    for a in axes:
        a.set_facecolor("#1a1d27")
        for spine in a.spines.values():
            spine.set_edgecolor("#444")
        a.tick_params(colors="#aaa", labelsize=8)
        a.xaxis.label.set_color("#aaa")
        a.yaxis.label.set_color("#aaa")
        a.title.set_color("white")


def make_figure(n_panels, height_per_panel=3.2, title=""):
    """Create a dark-themed figure with n vertically stacked panels."""
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(18, height_per_panel * n_panels),
        facecolor="#0e1117",
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=14, color="white", weight="bold")
    style_axes(axes)
    return fig, axes


def save_or_show(fig, output_path):
    """Save figure to file or display interactively."""
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  ✓ Saved → {output_path}")
    else:
        plt.show()
    plt.close()


def trim_to_shortest(*arrays):
    """Trim all arrays to the length of the shortest."""
    n = min(len(a) for a in arrays)
    return tuple(a[:n] for a in arrays)


# ── CLI helper ────────────────────────────────────────────────────────────────

def base_argparser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("audio_path", help="Path to audio file (mp3, wav, flac …)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save figure to this path (e.g. out.png). Omit → show window.")
    return parser