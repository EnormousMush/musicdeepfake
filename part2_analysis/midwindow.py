"""
30 秒中窗特征(人性线 H2 第二刀,2026-08-10):10s 窗看不见的三类"时间尺度"信号。

  漂移三件套  真人演奏 tempo 会呼吸(rubato/赶拖),AI 渲染多为石英钟——
              由 beat 序列导出局部 tempo 曲线(中值滤波去毛刺),量 cv/极差/趋势/最大跳变;
  呼吸缝隙    乐句之间的短静默:真人要换气/留白,AI 常无缝填满——
              RMS dB 低于阈值(min(-35dB, 中位-15dB))且 ≥150ms 的段;量个数/占比/时长/间隔规整度;
  响度弧线    30s 内音量走势的形状:线性斜率/曲率/峰值位置/极差——模板化编排的痕迹。

纯 librosa/numpy/scipy,零 ML。喂 h2_export 的 30s 共同窗(16k/mono/LUFS-23)。
"""
import numpy as np
import librosa
from scipy.signal import medfilt

HOP = 512
FRAME = 2048
GAP_MIN_S = 0.15


def analyze_midwindow(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    if len(y) < sr * 5:
        return {"error": "too_short"}
    out = {}
    t_per = HOP / sr

    # ---------- 响度弧线 ----------
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP)[0]
    db = 20.0 * np.log10(rms + 1e-8)
    win = max(1, int(round(1.0 / t_per)))            # 1s 平滑
    sm = np.convolve(db, np.ones(win) / win, mode="valid")
    x = np.arange(len(sm)) * t_per
    lin = np.polyfit(x, sm, 1)
    quad = np.polyfit(x, sm, 2)
    out["arc_slope_db_per_min"] = float(lin[0] * 60)
    out["arc_quad_db_per_s2"] = float(quad[0])
    out["arc_range_db"] = float(sm.max() - sm.min())
    out["arc_peak_pos"] = float(x[int(np.argmax(sm))] / x[-1])

    # ---------- 呼吸缝隙 ----------
    thr = min(-35.0, float(np.median(db)) - 15.0)
    below = db < thr
    runs = []                                        # (start_s, dur_s)
    i = 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            dur = (j - i) * t_per
            if dur >= GAP_MIN_S:
                runs.append((i * t_per, dur))
            i = j
        else:
            i += 1
    total_s = len(db) * t_per
    out["gap_count"] = float(len(runs))
    out["gap_total_ratio"] = float(sum(d for _, d in runs) / total_s)
    out["gap_mean_ms"] = float(1000 * np.mean([d for _, d in runs])) if runs else 0.0
    out["gap_max_ms"] = float(1000 * max(d for _, d in runs)) if runs else 0.0
    if len(runs) >= 3:
        iv = np.diff([s for s, _ in runs])
        out["gap_interval_cv"] = float(np.std(iv) / (np.mean(iv) + 1e-9))
    # 无缝隙 = 0 值有语义(填满),interval_cv 缺失合法(<3 个缝无"规整度"可言)

    # ---------- 漂移三件套 ----------
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    _tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP)
    bt = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    if len(bt) >= 9:                                 # ≥8 个 IBI 才谈漂移
        loc = 60.0 / np.diff(bt)                     # 局部 BPM 序列
        locm = medfilt(loc, 5)                       # 去 beat-tracker 毛刺
        mu = float(np.mean(locm))
        out["drift_cv"] = float(np.std(locm) / mu)
        out["drift_range_pct"] = float((locm.max() - locm.min()) / mu)
        out["drift_slope_bpm_per_min"] = float(np.polyfit(bt[1:], locm, 1)[0] * 60)
        step = np.abs(np.diff(locm)) / mu
        out["drift_max_step_pct"] = float(step.max()) if len(step) else 0.0
        out["n_beats_30s"] = float(len(bt))
    else:
        out["drift_missing"] = "too_few_beats"       # 合法缺失(自由节拍/环境声)
    return out
