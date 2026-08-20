"""频谱/亮度家族(10s 线)。公式自 part2_analysis/spectral.py 原样搬迁(2026-08-20 大修),
数值必须与 features62.csv 逐位一致 —— 只把"自己解码"改成"向 ctx 要",数学一行不动。
输出列名:stats.centroid_mean_hz 等(经 extract._flatten 加 "stats." 前缀,与历史 CSV 同名)。
"""
import numpy as np
import librosa


def run(ctx):
    y, sr, hop_length = ctx.y, ctx.SR, ctx.HOP

    D = librosa.stft(y, hop_length=hop_length)
    S_mag = np.abs(D)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length,
                                                    roll_percent=0.85)[0]
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    contrast_band_means = np.mean(spec_contrast, axis=1)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = freqs >= 8000
    hf_energy = np.sum(S_mag[hf_mask, :] ** 2)
    total_energy = np.sum(S_mag ** 2) + 1e-12
    hf_energy_ratio = hf_energy / total_energy

    mel_band_temporal_std = np.mean(np.std(mel_db, axis=1))

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
    return {"stats": stats}
