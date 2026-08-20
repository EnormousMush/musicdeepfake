"""FeatureContext:一个 clip 一次解码,所有共享中间量懒加载缓存(2026-08-20 大修核心)。

大修前:六个 10s 模块各自 librosa.load(解码 6 次)、HPSS 6 次、beat_track 3 次、
onset_detect 3 次 —— 参数完全一样,纯浪费。
大修后:六个家族共享本对象,每个中间量第一次被要时才算、算完缓存。

口径(与 2026-08 历史提取完全一致,一个参数不改):
  sr=22050, hop=512;HPSS margin=4;beat/onset 都在打击乐分量 y_perc 上,
  onset_detect 参数 pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.06, wait=10。
注:key 家族按历史口径自己以 sr=11025 重新解码(见 families/key.py),不走本对象。
"""
import numpy as np
import librosa


class FeatureContext:
    SR = 22050
    HOP = 512

    def __init__(self, path: str):
        self.path = path
        self._c = {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    # ---------- 波形 ----------
    @property
    def y(self):
        return self._get("y", lambda: librosa.load(self.path, sr=self.SR, mono=True)[0])

    @property
    def duration(self):
        return self._get("duration", lambda: librosa.get_duration(y=self.y, sr=self.SR))

    @property
    def hpss(self):
        """(y_harm, y_perc),margin=4 —— 全项目历史口径。"""
        return self._get("hpss", lambda: librosa.effects.hpss(self.y, margin=4))

    @property
    def y_harm(self):
        return self.hpss[0]

    @property
    def y_perc(self):
        return self.hpss[1]

    # ---------- 节拍 / onset(dynamics/rhythm/quantize 三家共用)----------
    @property
    def beats(self):
        """(tempo_raw, beat_frames),beat_track 于 y_perc。"""
        return self._get("beats", lambda: librosa.beat.beat_track(y=self.y_perc, sr=self.SR))

    @property
    def tempo(self):
        return float(np.atleast_1d(self.beats[0])[0])

    @property
    def beat_times(self):
        return self._get("beat_times",
                         lambda: librosa.frames_to_time(self.beats[1], sr=self.SR))

    @property
    def onset_frames(self):
        return self._get("onset_frames", lambda: librosa.onset.onset_detect(
            y=self.y_perc, sr=self.SR, units="frames",
            pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.06, wait=10))

    @property
    def onset_times(self):
        return self._get("onset_times",
                         lambda: librosa.frames_to_time(self.onset_frames, sr=self.SR))

    @property
    def onset_env_perc(self):
        return self._get("onset_env_perc", lambda: librosa.onset.onset_strength(
            y=self.y_perc, sr=self.SR, hop_length=self.HOP))
