"""乐器家族(占位,未启用)。Blase 方向5(2026-08-19 会议):特征面向 instruments 扩表。
落点在此:实现 run(ctx),在 registry.SETS 里注册新集,不动任何既有集。
候选方向(未预注册,仅备忘):乐器活跃度代理(谱带能量分布的时变)、
打击乐/和声乐配比细分(ctx.hpss 已有现成分量)。
"""


def run(ctx):
    raise NotImplementedError("instruments 家族未启用(Blase 方向5 扩表落点)")
