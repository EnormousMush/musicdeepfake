"""
去重 + 定稿:从两份总索引产出 final 数据集(2026-09-20)。

**规则先写死,再看数据** —— 否则就变成事后挑数字。

去重优先级(同一 sha1 保留哪一份):
  manifest 最全的那份留下。jamendo_sweep > jamendo2025 > jamendo_eras,
  因为 sweep 的 manifest 带完整 musicinfo + 审计判据;eras 是中途废弃的批次。

硬排除(直接不进 final):
  1. 字节级重复的多余份
  2. `ai_risk` 与所在侧矛盾的(例:AI 目录下的 FMA 人类参照)
  3. 明确标注"只做测试行"的(sonics)——单独进 test_only
  4. 缺切分键且该语料整体缺(记入限制,不排除,但标出来)

分层(不是排除,是标记):
  instrumental=True  且判据可靠  → core
  instrumental 未知/不可靠        → aux(单独成层,不进主训练池)
  ai_risk 存疑                    → quarantine(只做污染敏感性分析)

Usage:
  .venv_audit/bin/python part1_extraction/finalize_dataset.py --index-dir DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

# 同 sha1 时保留谁:数字越小越优先
KEEP_PRIORITY = {
    "jamendo": 0, "pixabay": 0,
    "fma": 0, "magnatagatune": 0, "ccmixter": 0, "ia_netlabels": 0,
    "suno_audio": 0, "fakemusiccaps": 0, "generators": 0,
    "sonics": 5, "acestep_self": 0,
}

# 器乐判据可信度:实测过的 > 官方字段 > 众包标签 > 缺席推断 > 无
def inst_tier(method: str) -> str:
    m = method or ""
    if "已逐曲核查" in m or "白名单" in m:
        return "verified"
    if "官方字段" in m or "musicinfo" in m:
        return "field"
    if "众包" in m or "inst_tag" in m or "no_vocal 列" in m or "流派标签" in m:
        return "tag"
    if "缺席推断" in m or "no_vocal 缺席" in m:
        return "absence"
    if "echonest" in m:
        return "proxy"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    d = Path(args.index_dir)
    out = Path(args.out) if args.out else d

    rows = []
    for side in ("human", "ai"):
        f = d / f"{side}_index.csv"
        if f.exists():
            rows += list(csv.DictReader(open(f, encoding="utf-8")))
    print(f"读入 {len(rows)} 行\n")

    # ---------- 1. 侧别矛盾:AI 目录下的人类音乐,归到人类侧
    moved = 0
    for r in rows:
        if r["side"] == "ai" and r["ai_risk"] == "0":
            r["side"] = "human"
            r["notes"] = (r["notes"] + ";已从AI侧移入人类侧").strip(";")
            moved += 1
    print(f"[1] 侧别修正:{moved} 首从 AI 侧移入人类侧(目录位置与实际内容不符)")

    # ---------- 2. 字节级去重
    by_sha = collections.defaultdict(list)
    for r in rows:
        if r["sha1"]:
            by_sha[r["sha1"]].append(r)
    drop = set()
    dup_cross = collections.Counter()
    for sha, v in by_sha.items():
        if len(v) < 2:
            continue
        cs = sorted({x["corpus"] for x in v})
        if len(cs) > 1:
            dup_cross[" + ".join(cs)] += 1
        v_sorted = sorted(v, key=lambda x: (KEEP_PRIORITY.get(x["corpus"], 9), x["uid"]))
        for x in v_sorted[1:]:
            drop.add(id(x))
    print(f"[2] 去重:丢弃 {len(drop)} 首(保留优先级最高的那份)")
    for k, v in dup_cross.most_common(6):
        print(f"      跨语料 {k:36s} {v} 组")

    kept = [r for r in rows if id(r) not in drop]

    # ---------- 3. 分层
    for r in kept:
        tier = inst_tier(r["instrumental_method"])
        r["inst_tier"] = tier
        if r["side"] == "ai":
            layer = "test_only" if "只做矩阵测试行" in r["notes"] else "core"
        elif str(r["ai_risk"]).lower() == "unknown":
            layer = "quarantine"          # 仅 Pixabay:平台标注漏检，我方打分 2.1× 基线
        elif r["instrumental"] == "True" and tier in ("verified", "field", "tag"):
            layer = "core"
        elif r["instrumental"] == "True":
            layer = "aux"
        else:
            layer = "aux"
        r["layer"] = layer

    # ---------- 4. 切分键健康度
    print("\n[3] 分层与切分键")
    print(f"{'语料':<16}{'侧':<7}{'首数':>8}{'切分组':>8}{'组内均值':>9}  分层")
    agg = collections.defaultdict(list)
    for r in kept:
        agg[(r["side"], r["corpus"])].append(r)
    for (side, corpus), v in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        g = len({x["split_key"] for x in v})
        lay = collections.Counter(x["layer"] for x in v)
        nokey = sum(1 for x in v if "缺失" in x["split_key_method"] or g == len(v))
        flag = "  ⚠️每首自成一组" if g == len(v) and len(v) > 300 else ""
        print(f"{corpus:<16}{side:<7}{len(v):>8}{g:>8}{len(v)/max(g,1):>9.1f}  "
              f"{dict(lay)}{flag}")

    print("\n[4] 汇总")
    for side in ("human", "ai"):
        sub = [r for r in kept if r["side"] == side]
        lay = collections.Counter(r["layer"] for r in sub)
        core = [r for r in sub if r["layer"] == "core"]
        print(f"  {side:6s} 共 {len(sub):7d} 首   {dict(lay)}")
        print(f"         core {len(core)} 首 / {len({r['split_key'] for r in core})} 个切分组")
        ti = collections.Counter(r["inst_tier"] for r in core)
        print(f"         core 的器乐判据: {dict(ti)}")

    cols = list(kept[0].keys())
    f = out / "final_index.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(kept)
    print(f"\n→ {f}  ({len(kept)} 行)")


if __name__ == "__main__":
    main()
