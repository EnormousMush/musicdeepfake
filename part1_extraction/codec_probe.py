"""
编码历史探测:两侧是不是存在"音乐之外"的系统性差异(2026-09-20)。

动机:实测发现人类侧几乎全是 mp3,而 AI 侧的 generators(我们自产 6,000 首)
是 flac、fakemusiccaps 是 wav。**鉴伪模型可以直接学「有 mp3 压缩痕迹 = 人类」,
根本不用听音乐。** 这和 SONICS 那次「有刀/无刀混装会教坏探针」是同一类错误,
只不过载体从 demucs 处理痕迹换成了编码历史。

污染让指标变差,捷径让指标**虚高**——而虚高的结果看起来像成功,更危险。

测三件事:
  1. **有损编码的频谱天花板**:mp3/aac 在高频有硬截止(常见 15–16 kHz),
     无损没有。量每首的实际截止频率。
  2. **共同规格之后还剩多少**:重采样到 16 kHz 只保留 0–8 kHz,理论上截止痕迹
     被削掉;但 mp3 在 8 kHz 以下也留痕(pre-echo、量化噪声)。所以要在
     **共同规格之后**再测一次可分性——这才是真正要回答的问题。
  3. **可分性**:拿最简单的频谱统计训一个分类器区分"人类语料 vs AI 语料",
     如果共同规格后仍然轻松分开,说明捷径还在。

Usage:
  .venv_audit/bin/python part1_extraction/codec_probe.py --n 120
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

CORP = Path("/Volumes/Seagate /honors_paper")
HUMAN = {
    "fma": CORP / "1_corpora_real/fma/fma_large",
    "jamendo_sweep": CORP / "1_corpora_real/jamendo_sweep/audio",
    "jamendo2025": CORP / "1_corpora_real/jamendo2025",
    "magnatagatune": CORP / "1_corpora_real/magnatagatune",
    "pixabay_all": CORP / "1_corpora_real/pixabay_all",
    "ccmixter": CORP / "1_corpora_real/ccmixter",
}
AI = {
    "generators": CORP / "2_corpora_ai/generators",
    "suno_audio": CORP / "2_corpora_ai/suno_audio",
    "fakemusiccaps": CORP / "2_corpora_ai/fakemusiccaps",
    "sonics": CORP / "2_corpora_ai/sonics",
}
EXTS = (".mp3", ".wav", ".flac", ".au", ".m4a", ".ogg")


def sample(d: Path, n, rng):
    fs = [p for p in d.rglob("*") if p.suffix.lower() in EXTS]
    rng.shuffle(fs)
    return fs[:n]


def _int(v):
    """ffprobe 对 flac/wav 的 bit_rate 会返回 'N/A',不能直接 int()。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def probe_meta(p: Path):
    """ffprobe 拿真实采样率/码率/编码器。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,bit_rate,channels",
             "-of", "default=nw=1", str(p)],
            capture_output=True, text=True, timeout=30)
        return dict(l.split("=", 1) for l in r.stdout.strip().splitlines() if "=" in l)
    except Exception:
        return {}


def cutoff_hz(y, sr, floor_db=-60.0):
    """频谱天花板:功率谱从峰值跌破 floor_db 且此后不再回升的频率。

    有损编码器在这里留硬截止(mp3 128k 常见 ~16 kHz),无损没有。
    """
    S = np.abs(librosa.stft(y, n_fft=4096)) ** 2
    psd = S.mean(axis=1)
    psd_db = 10 * np.log10(psd / (psd.max() + 1e-20) + 1e-20)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    above = np.where(psd_db > floor_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0


def spec_stats(y, sr):
    """共同规格之后的简单频谱统计——捷径探测器的输入。"""
    S = np.abs(librosa.stft(y, n_fft=1024))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=40)
    ml = librosa.power_to_db(mel + 1e-12)
    return np.concatenate([ml.mean(1), ml.std(1),
                           [librosa.feature.spectral_rolloff(S=S, sr=sr).mean(),
                            librosa.feature.spectral_flatness(S=S).mean()]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="每个语料抽多少首")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = random.Random(0)

    rows = []
    for side, group in [("human", HUMAN), ("ai", AI)]:
        for name, d in group.items():
            if not d.exists():
                print(f"  跳过 {name}(路径不存在)")
                continue
            files = sample(d, args.n, rng)
            print(f"[{side}/{name}] {len(files)} 首 …", flush=True)
            for p in files:
                meta = probe_meta(p)
                try:
                    # ① 原生采样率下量频谱天花板
                    y0, sr0 = librosa.load(str(p), sr=None, mono=True, duration=20)
                    cut = cutoff_hz(y0, sr0)
                    # ② 共同规格后的频谱统计
                    dur = len(y0) / sr0
                    y1, _ = librosa.load(str(p), sr=16000, mono=True,
                                         offset=max(0, dur / 2 - 5), duration=10)
                    if len(y1) < 16000 * 9:
                        continue
                    y1 = y1 / (np.sqrt(np.mean(y1 ** 2)) + 1e-9) * 10 ** (-23 / 20)
                    feat = spec_stats(y1, 16000)
                except Exception:
                    continue
                rows.append(dict(side=side, corpus=name, path=str(p),
                                 ext=p.suffix.lower().lstrip("."),
                                 codec=meta.get("codec_name", ""),
                                 native_sr=_int(meta.get("sample_rate")),
                                 bit_rate=_int(meta.get("bit_rate")),
                                 cutoff_hz=round(cut, 1), feat=feat.tolist()))

    print(f"\n共 {len(rows)} 首\n")
    print("=" * 74)
    print("① 原生格式与频谱天花板")
    print("=" * 74)
    print(f"{'语料':<16}{'侧':<7}{'格式':<20}{'原生sr':>9}{'截止中位':>11}")
    for name in list(HUMAN) + list(AI):
        sub = [r for r in rows if r["corpus"] == name]
        if not sub:
            continue
        ext = collections.Counter(f"{r['ext']}/{r['codec']}" for r in sub).most_common(1)[0][0]
        sr = int(np.median([r["native_sr"] for r in sub]))
        cut = np.median([r["cutoff_hz"] for r in sub])
        print(f"{name:<16}{sub[0]['side']:<7}{ext:<20}{sr:>9}{cut/1000:>9.1f}k")

    # ---- ② 共同规格之后,还能不能靠频谱把两侧分开
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    X = np.array([r["feat"] for r in rows])
    y = np.array([1 if r["side"] == "ai" else 0 for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    acc = cross_val_score(LogisticRegression(max_iter=4000, C=0.1),
                          (X - mu) / sd, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    base = max(y.mean(), 1 - y.mean())
    print("\n" + "=" * 74)
    print("② 共同规格(10s/16k/mono/-23LUFS)之后,只用简单频谱统计分两侧")
    print("=" * 74)
    print(f"   5 折准确率 {acc:.1%}   (多数类基线 {base:.1%})")
    if acc > base + 0.15:
        print("   ⚠️ 仍能轻松分开 —— 捷径没被共同规格消掉,必须做对称转码")
    else:
        print("   ✅ 接近基线 —— 共同规格已经削掉了大部分格式痕迹")

    out = Path(args.out) if args.out else Path("codec_probe.json")
    out.write_text(json.dumps(
        dict(n=len(rows), sep_acc=float(acc), baseline=float(base),
             by_corpus={n: dict(
                 ext=collections.Counter(r["ext"] for r in rows if r["corpus"] == n).most_common(),
                 cutoff_median=float(np.median([r["cutoff_hz"] for r in rows if r["corpus"] == n])))
                 for n in set(r["corpus"] for r in rows)}),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
