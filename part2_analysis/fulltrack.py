"""
全长结构特征(人性线 H2 第三刀,2026-08-10):只有整首歌才存在的信号。suno×jamendo 双雄
(FMA 无全长缺席)。全部纯 DSP,零 ML;原盘直读(无 LUFS——所有响度特征取相对量/斜率)。

  段落结构   chroma+MFCC 堆叠 → 余弦自相似矩阵(降到 ~2fps)→ 棋盘核新奇度 → 峰=段落边界:
             每分钟段落数 / 段长CV / 段长中位 / 方正度(段长≈中位整数倍的占比);
  副歌复制   时滞对角条纹:每个滞后 ℓ 的对角线相似度取 8s 滑窗最大值,全滞后取最大 =
             chorus_copy_max(真人复演必有差异,AI 渲染近像素复制)+ 高相似条纹最长时长;
  淡出形状   末 20s RMS dB:线性斜率 / 拟合R² / 突兀度(末1s−全曲中位)/ 尾部静默时长;
  长程漂移   全曲 beat 序列局部 BPM 曲线的 cv(30s 窗量不到的分钟级 rubato);
  立体声宽度 side/mid 能量比(dB)逐秒:均值 / std / L-R 相关。

时长上限 480s(8min 截断);duration_s 落盘作控制变量(结构计数均按每分钟归一)。
"""
import numpy as np
import librosa
from scipy.signal import find_peaks

SR = 16000
MAX_DUR = 480.0
HOP = 2048                      # ~7.8 fps 特征帧
AGG = 4                         # 聚 4 帧 → ~2 fps
KERNEL_S = 8.0                  # 棋盘核半宽(秒)


def _ssm_features(y, sr, out):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    F = np.vstack([chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9),
                   mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-9)])
    n = F.shape[1] // AGG
    if n < 30:
        out["struct_missing"] = "too_short"
        return
    Fa = F[:, : n * AGG].reshape(F.shape[0], n, AGG).mean(axis=2)
    Fa = Fa / (np.linalg.norm(Fa, axis=0, keepdims=True) + 1e-9)
    S = Fa.T @ Fa                                  # 余弦 SSM,~2fps
    fps = sr / HOP / AGG
    k = max(4, int(round(KERNEL_S * fps)))

    # ---- 新奇度(棋盘核沿对角线) ----
    nov = np.zeros(n)
    for i in range(k, n - k):
        a = S[i - k:i, i - k:i].mean()
        b = S[i:i + k, i:i + k].mean()
        c = S[i - k:i, i:i + k].mean()
        nov[i] = a + b - 2 * c
    nov = (nov - nov.min()) / (np.ptp(nov) + 1e-9)
    peaks, _ = find_peaks(nov, distance=int(8 * fps), prominence=0.1)
    bounds = np.r_[0, peaks, n]
    lens = np.diff(bounds) / fps                   # 段长(秒)
    dur_min = n / fps / 60.0
    out["sec_count_per_min"] = float(len(peaks) / dur_min)
    if len(lens) >= 3:
        out["sec_len_cv"] = float(np.std(lens) / (np.mean(lens) + 1e-9))
        med = float(np.median(lens))
        out["sec_len_median_s"] = med
        ratio = lens / (med + 1e-9)
        out["sec_squareness"] = float(np.mean(np.abs(ratio - np.round(ratio)) < 0.15))

    # ---- 副歌复制(时滞条纹) ----
    w = int(round(8 * fps))                        # 8s 滑窗
    best, best_len = 0.0, 0.0
    for lag in range(int(8 * fps), n - w):
        d = np.diagonal(S, offset=lag)
        if len(d) < w:
            continue
        ma = np.convolve(d, np.ones(w) / w, mode="valid")
        m = float(ma.max())
        if m > best:
            best = m
        hi = ma > 0.90
        if hi.any():
            runs = np.diff(np.flatnonzero(np.diff(np.r_[0, hi.view(np.int8), 0])))[::2]
            best_len = max(best_len, float(runs.max() + w) / fps)
    out["chorus_copy_max"] = best
    out["chorus_copy_len_s"] = best_len


def analyze_fulltrack(path):
    ys, sr = librosa.load(path, sr=SR, mono=False, duration=MAX_DUR)
    out = {}
    if ys.ndim == 2 and ys.shape[0] == 2:
        L, R = ys[0], ys[1]
        y = ys.mean(axis=0)
        # ---- 立体声宽度 ----
        mid, side = (L + R) / 2, (L - R) / 2
        blk = sr                                    # 1s 块
        nb = len(y) // blk
        if nb >= 10:
            em = np.array([np.mean(mid[i*blk:(i+1)*blk] ** 2) for i in range(nb)])
            es = np.array([np.mean(side[i*blk:(i+1)*blk] ** 2) for i in range(nb)])
            wdb = 10 * np.log10((es + 1e-12) / (em + 1e-12))
            out["width_mean_db"] = float(np.mean(wdb))
            out["width_std_db"] = float(np.std(wdb))
            out["lr_corr"] = float(np.corrcoef(L[: nb*blk], R[: nb*blk])[0, 1])
    else:
        y = ys if ys.ndim == 1 else ys.mean(axis=0)
        out["stereo_missing"] = "mono_source"
    dur = len(y) / SR
    out["duration_s"] = float(dur)
    if dur < 45:
        out["error"] = "too_short"
        return out

    # ---- 淡出形状(末 20s)----
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    db = 20 * np.log10(rms + 1e-8)
    t_per = 512 / SR
    smooth = max(1, int(1.0 / t_per))
    sm = np.convolve(db, np.ones(smooth) / smooth, mode="valid")
    med_db = float(np.median(sm))
    tail = sm[-int(20 / t_per):]
    x = np.arange(len(tail)) * t_per
    sl, b = np.polyfit(x, tail, 1)
    pred = sl * x + b
    ssr = np.sum((tail - pred) ** 2); sst = np.sum((tail - tail.mean()) ** 2) + 1e-9
    out["fade_slope_db_per_s"] = float(sl)
    out["fade_r2"] = float(1 - ssr / sst)
    out["end_abruptness_db"] = float(np.mean(sm[-int(1 / t_per):]) - med_db)
    below = sm < med_db - 25
    ts = 0
    for v in below[::-1]:
        if v: ts += 1
        else: break
    out["tail_silence_s"] = float(ts * t_per)

    # ---- 长程漂移 ----
    oenv = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    _t, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=512)
    bt = librosa.frames_to_time(beats, sr=SR, hop_length=512)
    if len(bt) >= 30:
        loc = 60.0 / np.diff(bt)
        from scipy.signal import medfilt
        locm = medfilt(loc, 9)
        out["drift_cv_full"] = float(np.std(locm) / (np.mean(locm) + 1e-9))
    # ---- 结构 ----
    _ssm_features(y, SR, out)
    return out
