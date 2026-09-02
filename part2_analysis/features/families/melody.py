"""旋律家族(10s 线,2026-09-03 启用;Blase 方向5 / 第一幕 v2 扩表第一刀)。

零 ML 铁律:全部纯 DSP + 统计。音高轨来自 librosa.pyin(概率 YIN,经典信号处理,
无学习参数);其余全是对 f0 序列的算术。

核心特征 = **音准量化度**,与 quantize.py 的时值量化度逐项对称:
  时值线:onset 对节拍网格的偏差(ms)→ quantization_score
  音准线:f0 对 12 平均律网格的偏差(cents)→ pitch_quant_score
先扣掉全局调音偏移(A≠440 不算"不准"),再量残差——人手/人声的音准游移 vs 合成器的死准。

其余三组(全部按 voiced 帧计算):
  微动:相邻帧音高变化(cents)——颤音/游移 vs 平直;
  轮廓:音符切分后的音程分布(级进/跳进/音域/音符速率/时值 cv);
  音阶与重复:最佳大小调的音阶内占比、音级熵、音程 n-gram 重复率(旋律重复率)。

输入:ctx.y_harm(HPSS 谐波分量,margin=4,历史口径),sr=22050,hop=512,
pyin 范围 C2–C7。voiced 占比 <10% 或音符 <4 个 → 返回 error(与 quantize 家族同风格)。
列名前缀由 registry 的 tag 决定;本模块不返回任何数组(SKIP_KEYS 无需增项)。
"""
import numpy as np
import librosa

FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")
FRAME = 2048
MIN_VOICED_RATIO = 0.10
MIN_NOTES = 4
MIN_NOTE_FRAMES = 3          # 音符至少 3 帧(≈70ms)才算一个音


def _circular_mean_cents(dev):
    """偏差在 [-50, 50) cents 上是周期量(周期 100),用圆均值估全局调音偏移。"""
    ang = dev / 100.0 * 2 * np.pi
    m = np.angle(np.mean(np.exp(1j * ang)))
    return float(m / (2 * np.pi) * 100.0)


def _wrap_cents(dev):
    return (dev + 50.0) % 100.0 - 50.0


def _segment_notes(midi_round, voiced):
    """把逐帧四舍五入的 MIDI 序列切成音符 (pitch, n_frames);unvoiced 断开。"""
    notes = []
    cur, n = None, 0
    for m, v in zip(midi_round, voiced):
        if not v:
            if cur is not None and n >= MIN_NOTE_FRAMES:
                notes.append((cur, n))
            cur, n = None, 0
            continue
        if cur is None or m != cur:
            if cur is not None and n >= MIN_NOTE_FRAMES:
                notes.append((cur, n))
            cur, n = m, 1
        else:
            n += 1
    if cur is not None and n >= MIN_NOTE_FRAMES:
        notes.append((cur, n))
    return notes


# 24 个大小调的音阶模板(半音集合)
_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]


def _scale_fit(pc_hist):
    """最佳大小调下,音阶内音级占总时长的比例。"""
    best = 0.0
    for tonic in range(12):
        for scale in (_MAJOR, _MINOR):
            idx = [(tonic + s) % 12 for s in scale]
            best = max(best, float(pc_hist[idx].sum()))
    return best


def _ngram_repeat_ratio(seq, n):
    """长度 n 的 n-gram 中,出现 ≥2 次的 n-gram 所占的 token 比例。"""
    if len(seq) < n + 1:
        return 0.0
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    rep = sum(1 for g in grams if counts[g] >= 2)
    return float(rep / len(grams))


def run(ctx):
    sr, hop = ctx.SR, ctx.HOP
    f0, voiced, vprob = librosa.pyin(ctx.y_harm, fmin=FMIN, fmax=FMAX, sr=sr,
                                     frame_length=FRAME, hop_length=hop)
    voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
    n_frames = len(f0)
    voiced_ratio = float(voiced.mean()) if n_frames else 0.0
    if voiced_ratio < MIN_VOICED_RATIO:
        return {"error": f"voiced ratio {voiced_ratio:.2f} < {MIN_VOICED_RATIO} (no melody)."}

    midi = librosa.hz_to_midi(f0[voiced])                     # 连续 MIDI
    midi_full = np.where(voiced, librosa.hz_to_midi(np.where(voiced, f0, 1.0)), np.nan)

    # ---------- 音准量化度(对称于时值量化)----------
    dev_raw = _wrap_cents((midi - np.round(midi)) * 100.0)   # 对网格的偏差,cents
    tuning_offset = _circular_mean_cents(dev_raw)             # 全局调音偏移
    dev = _wrap_cents(dev_raw - tuning_offset)                # 扣掉调音后的残差
    abs_dev = np.abs(dev)
    pitch_quant_score = float(max(0.0, 100.0 * (1.0 - abs_dev.mean() / 50.0)))

    # ---------- 微动:相邻 voiced 帧的音高变化 ----------
    cents_full = midi_full * 100.0
    d = np.diff(cents_full)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        d = np.zeros(1)
    abs_d = np.abs(d)

    # ---------- 轮廓:音符切分 ----------
    midi_round_full = np.where(voiced, np.round(np.nan_to_num(midi_full, nan=0.0)), -1).astype(int)
    notes = _segment_notes(midi_round_full, voiced)
    if len(notes) < MIN_NOTES:
        return {"error": f"only {len(notes)} notes segmented (< {MIN_NOTES})."}
    pitches = np.array([p for p, _ in notes], dtype=float)
    durs = np.array([n for _, n in notes], dtype=float) * hop / sr
    intervals = np.diff(pitches)
    abs_int = np.abs(intervals)
    duration_s = n_frames * hop / sr

    # ---------- 音阶与重复 ----------
    pc = (pitches.astype(int) % 12)
    pc_hist = np.bincount(pc, weights=durs, minlength=12)
    pc_hist = pc_hist / max(pc_hist.sum(), 1e-9)
    pc_entropy = float(-(pc_hist[pc_hist > 0] * np.log2(pc_hist[pc_hist > 0])).sum())
    int_seq = [int(round(x)) for x in intervals]

    return {
        # 覆盖
        "voiced_ratio":        round(voiced_ratio, 4),
        "voiced_prob_mean":    round(float(np.nanmean(vprob)), 4),
        # 音准量化度
        "tuning_offset_cents": round(tuning_offset, 2),
        "pitch_dev_mean_cents": round(float(abs_dev.mean()), 2),
        "pitch_dev_std_cents":  round(float(dev.std()), 2),
        "pitch_locked_ratio":   round(float((abs_dev <= 10).mean()), 4),   # ±10 cents 内
        "pitch_quant_score":    round(pitch_quant_score, 1),
        # 微动
        "pitch_jitter_cents":   round(float(abs_d.mean()), 2),
        "pitch_move_ratio":     round(float((abs_d > 5).mean()), 4),
        "pitch_jitter_std":     round(float(abs_d.std()), 2),
        # 轮廓
        "num_notes":            int(len(notes)),
        "note_rate_hz":         round(float(len(notes) / duration_s), 3),
        "note_dur_cv":          round(float(durs.std() / max(durs.mean(), 1e-9)), 4),
        "interval_mean_abs":    round(float(abs_int.mean()), 3),
        "step_ratio":           round(float((abs_int <= 2).mean()), 4),   # 级进
        "leap_ratio":           round(float((abs_int > 4).mean()), 4),    # 跳进
        "repeat_note_ratio":    round(float((abs_int == 0).mean()), 4),   # 同音重复
        "pitch_range_semi":     round(float(pitches.max() - pitches.min()), 1),
        # 音阶与重复
        "scale_fit":            round(_scale_fit(pc_hist), 4),
        "pc_entropy_bits":      round(pc_entropy, 4),
        "interval_bigram_rep":  round(_ngram_repeat_ratio(int_seq, 2), 4),
        "interval_trigram_rep": round(_ngram_repeat_ratio(int_seq, 3), 4),
    }
