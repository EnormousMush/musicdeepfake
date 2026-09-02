"""旋律家族合成音频单元测试(2026-09-03)。不依赖任何数据盘:全部用 numpy 合成。

四个已知答案的用例:
  1. 纯 A4 正弦:音准量化度应≈100,jitter≈0,单音 → 音符数不足报 error(合理);
  2. 上行音阶 C4→C5(每音 0.5s):音符数 8,级进比例≈1,scale_fit≈1,量化度高;
  3. 同一音阶整体 +25 cents:tuning_offset≈+25,扣偏移后量化度仍高(调音≠不准);
  4. 带 6Hz/±50cents 颤音的音阶:jitter 明显大于用例 2,quant_score 明显低于用例 2;
  5. 白噪声:voiced 太少 → error。
Usage:  <repo>/.venv/bin/python part2_analysis/features/melody_test.py
"""
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from part2_analysis.features.context import FeatureContext
from part2_analysis.features.families import melody

SR = 16000
DUR = 10.0


def _tone_seq(midis, seg_s, cents_offset=0.0, vibrato_hz=0.0, vibrato_cents=0.0):
    t = np.arange(int(SR * DUR)) / SR
    y = np.zeros_like(t)
    n = len(midis)
    for i, m in enumerate(midis):
        a, b = int(i * seg_s * SR), int(min((i + 1) * seg_s, DUR) * SR)
        if a >= len(t):
            break
        tt = t[a:b]
        cents = cents_offset + vibrato_cents * np.sin(2 * np.pi * vibrato_hz * tt)
        f = 440.0 * 2 ** ((m - 69 + cents / 100.0) / 12.0)
        # 相位连续积分,避免颤音时的相位跳变
        phase = 2 * np.pi * np.cumsum(f) / SR
        env = np.ones_like(tt); k = int(0.01 * SR)
        env[:k] = np.linspace(0, 1, k); env[-k:] = np.linspace(1, 0, k)
        y[a:b] = 0.3 * np.sin(phase) * env
    return y


def _run(y):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, SR)
        path = f.name
    try:
        return melody.run(FeatureContext(path))
    finally:
        os.unlink(path)


def main():
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        fails += 0 if cond else 1

    # 1. 纯 A4
    r1 = _run(_tone_seq([69], DUR))
    check("单音报 error(音符不足)", "error" in r1, str(r1)[:60])

    # 2. 上行音阶
    scale = [60, 62, 64, 65, 67, 69, 71, 72]
    r2 = _run(_tone_seq(scale, 1.0))
    check("音阶无 error", "error" not in r2, str(r2)[:60])
    if "error" not in r2:
        check("音符数 ≈ 8", 7 <= r2["num_notes"] <= 9, r2["num_notes"])
        check("量化度高", r2["pitch_quant_score"] > 85, r2["pitch_quant_score"])
        check("级进比例 ≈ 1", r2["step_ratio"] > 0.85, r2["step_ratio"])
        check("scale_fit ≈ 1", r2["scale_fit"] > 0.95, r2["scale_fit"])
        check("jitter 小", r2["pitch_jitter_cents"] < 8, r2["pitch_jitter_cents"])

    # 3. 整体偏 +25 cents:调音偏移被识别,残差量化度仍高
    r3 = _run(_tone_seq(scale, 1.0, cents_offset=25.0))
    check("调音偏移 ≈ +25", "error" not in r3 and abs(r3["tuning_offset_cents"] - 25) < 6,
          r3.get("tuning_offset_cents"))
    check("扣偏移后量化度仍高", "error" not in r3 and r3["pitch_quant_score"] > 85,
          r3.get("pitch_quant_score"))

    # 4. 颤音:jitter 大、量化度低
    r4 = _run(_tone_seq(scale, 1.0, vibrato_hz=6.0, vibrato_cents=50.0))
    if "error" not in r4 and "error" not in r2:
        check("颤音 jitter > 音阶 jitter ×3", r4["pitch_jitter_cents"] > 3 * r2["pitch_jitter_cents"],
              f"{r4['pitch_jitter_cents']} vs {r2['pitch_jitter_cents']}")
        check("颤音量化度 < 音阶量化度 − 15", r4["pitch_quant_score"] < r2["pitch_quant_score"] - 15,
              f"{r4['pitch_quant_score']} vs {r2['pitch_quant_score']}")
    else:
        check("颤音用例无 error", False, str(r4)[:60])

    # 5. 白噪声
    rng = np.random.default_rng(0)
    r5 = _run(0.1 * rng.standard_normal(int(SR * DUR)))
    check("白噪声报 error", "error" in r5, str(r5)[:60])

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAIL(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
