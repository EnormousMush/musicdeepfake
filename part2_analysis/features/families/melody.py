"""旋律家族(占位,未启用)。Blase 方向5(2026-08-19 会议):特征面向 melody 扩表。
落点在此:实现 run(ctx),在 registry.SETS 里注册新集(如 "h1_melody"),
不动任何既有集 —— 历史列名与数值永远可复现。
候选方向(未预注册,仅备忘):音高轮廓统计(pyin f0 的活跃度/音程分布)、
旋律重复率、音阶内外音比例。
"""


def run(ctx):
    raise NotImplementedError("melody 家族未启用(Blase 方向5 扩表落点)")
