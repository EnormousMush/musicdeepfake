"""特征集注册表(2026-08-20 大修):谁属于哪个集、输入规格是什么,一处说清。

三个特征集(与历史 CSV 一一对应):
  h1_10s   62 手工特征 10s 线(features62 / featuresJam / featuresSuno10mid / f62on30 同款)
  h2_mid   30s 中窗三类(featuresH2mid:响度弧线/呼吸缝隙/漂移三件套)
  h2_full  全长结构(featuresH2full:段落/副歌复制/淡出/长程漂移/立体声宽度)

家族 mode:
  ctx  — run(FeatureContext):共享中间量(解码/HPSS/beat/onset 只算一次)
  path — run(path):模块自己解码(H2 两集单模块,无共享收益;key 是历史 11025 口径)
"""
from .families import spectral, timbral, dynamics, rhythm, quantize, key
from .families import midwindow, fulltrack

# (tag, module.run, mode);tag 用于错误列名 f"{tag}_error",与历史 CSV 一致
SETS = {
    "h1_10s": [
        ("spec", spectral.run, "ctx"),
        ("timb", timbral.run, "ctx"),
        ("dyn",  dynamics.run, "ctx"),
        ("rhy",  rhythm.run, "ctx"),
        ("qnt",  quantize.run, "ctx"),
        ("key",  key.run, "path_via_ctx"),   # 拿 ctx 只为取 path(历史 11025 口径自己解码)
    ],
    "h2_mid":  [("mw", midwindow.run, "path")],
    "h2_full": [("ft", fulltrack.run, "path")],
}

# 输入规格声明(extract.py 开跑前抽样断言;数据初硬性统一从文档变成代码)
SPEC = {
    "h1_10s":  dict(window_s=10.0, tol=0.2, note="common-spec 10s(16k/mono/LUFS-23;模块内重采 22050)"),
    "h2_mid":  dict(window_s=30.0, tol=0.2, note="h2_export 30s 窗(16k/mono/LUFS-23)"),
    "h2_full": dict(window_s=None, min_s=45.0, note="全长原盘(≥45s;fulltrack 内部 16k 重采,480s 截断)"),
}

# 展开器黑名单:模块返回的非特征键(数组/中间量/基础设施),一律不落盘。
# 2026-08-20 冗余审计教训:旧版漏了 "ibis" → features62.csv 泄进 21 列事故列。
SKIP_KEYS = {"y", "sr", "S", "S_db", "mel_db", "times", "beat_times", "onset_times",
             "onset_env", "rms", "rms_db", "chroma", "mfcc", "contrast", "grid",
             "ibis", "duration", "hop_length", "frame_times", "oe_times",
             "times_samples", "spec_centroid", "spec_rolloff", "spec_bandwidth",
             "zcr", "spec_flatness", "rms_harm", "rms_perc", "crest"}
