"""
频谱包络匹配 — "亮度"假设的终审(升级版带宽匹配)。

把每首歌的长期平均频谱形状 EQ 到同一条全局目标曲线:做完后所有 clip 的平均音色一致,
"谁更亮/频谱倾斜"这个维度不再携带类别信息;但每个频段内的时间纹理(瞬态/混响/噪底/
解码痕迹)全部保留。然后重抽特征、逐层重探针,与原始对比。

  - EER 基本不动 -> 探针不靠平均音色,"亮度"假设彻底排除,信号在时频纹理里
  - EER 大幅上升 -> 近 0% 主要由亮度/频谱倾斜代差驱动,频谱匹配应进标准预处理

实现:STFT 幅度谱 -> 每 clip 的长期平均谱(mel 域平滑成包络)-> 增益曲线 = 目标包络/自身包络
(限幅 ±24dB)-> 幅度乘增益、相位不动 -> iSTFT。目标包络 = 两类各采样 300 首的全局平均
(固定种子,缓存到 specmatch_target.npy,断点续跑一致)。

Usage (server, venv active, GPU1; 目标包络 ~10min CPU + MERT 重抽 1-2h GPU):
  python diagnostics/spectral_match.py --data-dir data_store/subset_export_round1 --encoder mert
  加 --limit 20 先干跑。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf          # noqa: F401 (probe helpers import it)
from diagnostics.bandwidth_ablation import probe_all_layers, load_cached

N_FFT, HOP = 1024, 256
N_MEL = 64          # 包络平滑的粗糙度:只整形宏观音色,不抹细结构
GAIN_LIM_DB = 24.0


def _mel_mats(sr):
    import librosa
    M = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MEL)   # [mel, bin]
    Mn = M / (M.sum(axis=1, keepdims=True) + 1e-12)             # 行归一(mel <- bin 平均)
    P = M / (M.sum(axis=0, keepdims=True) + 1e-12)              # 列归一(bin <- mel 插值)
    return Mn, P


def clip_envelope(wav, sr):
    """长期平均幅度谱 -> mel 域包络 [N_MEL](log 域)。"""
    import librosa
    S = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP))  # [bin, T]
    env = S.mean(axis=1)                                        # [bin]
    Mn, _ = _mel_mats(sr)
    return np.log(Mn @ env + 1e-10)                             # [mel]


def match_envelope(wav, sr, target_logmel):
    """把 wav 的长期包络 EQ 到 target;返回时域信号(相位保留)。"""
    import librosa
    S = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)          # complex [bin, T]
    mag = np.abs(S)
    env_bin = mag.mean(axis=1)                                  # [bin]
    Mn, P = _mel_mats(sr)
    env_mel = np.log(Mn @ env_bin + 1e-10)
    gain_mel = target_logmel - env_mel                          # log 域差
    lim = GAIN_LIM_DB / 20.0 * np.log(10.0)
    gain_mel = np.clip(gain_mel, -lim, lim)
    gain_bin = np.exp(P.T @ gain_mel)                           # [bin],插值回线性频点
    y = librosa.istft(S * gain_bin[:, None], hop_length=HOP, length=len(wav))
    return np.asarray(y, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--target-n", type=int, default=300, help="每类采样 N 首估计全局目标包络")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from encoders.ssl import SSLEncoder

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))

    # ---- 1) 全局目标包络(两类各 target_n 首平均;缓存) ----
    tpath = data_dir / "specmatch_target.npy"
    if tpath.exists():
        target = np.load(tpath)
        print(f"目标包络:载入缓存 {tpath}")
    else:
        rng = np.random.default_rng(0)
        by = {0: [r for r in rows if int(r["label"]) == 0],
              1: [r for r in rows if int(r["label"]) == 1]}
        sample = list(rng.choice(by[0], min(args.target_n, len(by[0])), replace=False)) + \
                 list(rng.choice(by[1], min(args.target_n, len(by[1])), replace=False))
        envs = []
        t0 = time.time()
        for i, r in enumerate(sample, 1):
            wav, srate = sf.read(data_dir / r["rel_path"])
            envs.append(clip_envelope(np.asarray(wav, dtype=np.float32), srate))
            if i % 100 == 0:
                print(f"  目标包络 {i}/{len(sample)} ({time.time()-t0:.0f}s)")
        target = np.mean(envs, axis=0)
        np.save(tpath, target)
        print(f"目标包络:{len(sample)} 首平均,存 {tpath}")

    if args.limit:
        keep, seen = [], {0: 0, 1: 0}
        for r in rows:
            lb = int(r["label"])
            if seen[lb] < args.limit:
                keep.append(r); seen[lb] += 1
        rows = keep
        print(f"Dry-run: {len(rows)} clips")

    enc = SSLEncoder(args.encoder, device=args.device)
    print(f"Encoder: {args.encoder} on {enc.device} | 频谱包络匹配(增益限幅 ±{GAIN_LIM_DB:.0f}dB)")

    cache = data_dir / "features_specmatch" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)

    # ---- 2) 匹配 + 抽特征 ----
    F, y, sp, failures = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cpath = cache / f"{r['audio_id']}.npy"
        try:
            if cpath.exists():
                f = np.load(cpath)
            else:
                wav, srate = sf.read(data_dir / r["rel_path"])
                wav = match_envelope(np.asarray(wav, dtype=np.float32), srate, target)
                f = enc.encode_all_layers(wav, srate)
                np.save(cpath, f)
            F.append(f); y.append(int(r["label"])); sp.append(r["split"])
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 100 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)")

    F = np.stack(F); y = np.array(y); sp = np.array(sp)
    sm = probe_all_layers(F, y, sp)

    Fo, yo, spo = load_cached(rows, data_dir / "features" / args.encoder)
    orig = probe_all_layers(Fo, yo, spo) if Fo is not None else None

    print("\n=== 逐层 EER:原始 vs 频谱包络匹配后 ===")
    print(f"{'layer':>5} {'orig val':>9} {'orig test':>10} {'match val':>10} {'match test':>11}")
    for L, ev, et in sm:
        if orig:
            _, oev, oet = orig[L]
            print(f"{L:>5} {oev*100:>8.2f}% {oet*100:>9.2f}% {ev*100:>9.2f}% {et*100:>10.2f}%")
        else:
            print(f"{L:>5} {'--':>9} {'--':>10} {ev*100:>9.2f}% {et*100:>10.2f}%")

    best = min(sm, key=lambda p: p[1] if np.isfinite(p[1]) else 9)
    print(f"\n匹配后最佳层(val):layer {best[0]}  val {best[1]*100:.2f}%  test {best[2]*100:.2f}%")
    print("判读:match test 相比 orig test 大幅上升 -> 亮度/倾斜是主凶;基本不动 -> 亮度假设彻底排除,信号在时频纹理。")
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")


if __name__ == "__main__":
    main()
