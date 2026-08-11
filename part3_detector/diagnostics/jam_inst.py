"""jamendo 纯 instrumental 名单(人声成分核查,2026-08-11)。

背景:jamendo2025 采集时未加人声过滤,而 suno/fma 均为 instrumental。Jamendo API musicinfo
逐曲核查(2943/2943):instrumental 2050 / vocal 723 / 未标 170。手工特征侧所有正式结论
已改用 inst 子集;SSL 侧诊断脚本加 --inst-only 对齐口径,避免"AI vs 人"混进"器乐 vs 人声"。

名单 jamendo_instrumental.txt 由 Mac 侧生成(JSON 标签 × manifest,按 track_id 首次出现去重,
57 首双 genre 曲目只留一份),随 diagnostics/ 一同 rsync 上服务器,无外部依赖。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIST_PATH = os.path.join(HERE, "jamendo_instrumental.txt")


def instrumental_ids(path=None):
    """返回 {audio_id} 集合;名单文件缺失返回 None(调用方视为不过滤并告警)。"""
    p = path or LIST_PATH
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {ln.strip() for ln in f if ln.strip()}


def filter_rows(rows, inst_ids, source_key="source", id_key="audio_id", target="jamendo"):
    """只对 target 源生效:非 target 全留,target 只留名单内。"""
    if not inst_ids:
        return rows
    return [r for r in rows if r[source_key] != target or r[id_key] in inst_ids]
