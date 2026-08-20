"""调性估计家族(10s 线)。公式自 part2_analysis/key.py 原样搬迁(2026-08-20 大修)。
历史怪癖(保留不改,改了数字对不上):本家族在 sr=11025 上自己解码 + 自己 HPSS,
不共享 ctx 的 22050 波形 —— 因为"先载 22050 再降采样"与"直接以 11025 解码"结果有微差,
为对拍一致维持原样。这是 10s 线唯一的第二次解码。
输出列:best_key / best_corr / alt_key / alt_corr(alt 缺失合法 = 调性明确)。
"""
import numpy as np
import librosa

_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def run(ctx):
    y, sr = librosa.load(ctx.path, sr=11025, mono=True)
    y_harmonic, _ = librosa.effects.hpss(y, margin=4)
    chromograph = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=24)

    chroma_vals = [float(np.sum(chromograph[i])) for i in range(12)]
    keyfreqs = {_NOTE_NAMES[i]: chroma_vals[i] for i in range(12)}

    key_dict = {}
    for i in range(12):
        key_test = [keyfreqs[_NOTE_NAMES[(i + m) % 12]] for m in range(12)]
        maj_corr = round(float(np.corrcoef(_MAJOR_PROFILE, key_test)[1, 0]), 3)
        min_corr = round(float(np.corrcoef(_MINOR_PROFILE, key_test)[1, 0]), 3)
        key_dict[f"{_NOTE_NAMES[i]} major"] = maj_corr
        key_dict[f"{_NOTE_NAMES[i]} minor"] = min_corr

    best_key = max(key_dict, key=key_dict.get)
    best_corr = key_dict[best_key]

    alt_key, alt_corr = None, None
    for key_name, corr in key_dict.items():
        if corr > best_corr * 0.9 and corr != best_corr:
            alt_key = key_name
            alt_corr = corr

    out = {"best_key": str(best_key), "best_corr": float(best_corr)}
    if alt_key is not None:
        out["alt_key"] = str(alt_key)
        out["alt_corr"] = float(alt_corr)
    return out
