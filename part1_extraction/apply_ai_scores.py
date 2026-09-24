"""
把 MERT 打分结果回写进 final_index.csv(2026-09-23,路 B)。

对 pixabay 全量 4,891 首:低于阈值 → 出 quarantine 进 core;高于阈值 → 留 quarantine。
阈值沿用 audit_ai.py 的定义:留出人类参照的 p95(= 5% 误报点)。

局限(必须写进 ai_risk_method):这把尺子在 5% 误报点只检出 56.7% 的已知 AI,
踢掉的是"明显像我们见过的 AI"那部分,漏掉的仍在 core 里。

Usage:
  .venv_audit/bin/python part1_extraction/apply_ai_scores.py \
      --index _INDEX/final_index.csv --scores px_full/ai_scores.csv --corpus pixabay
"""
from __future__ import annotations

import argparse
import collections
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--scores", required=True, help="audit_ai.py 输出的逐曲分数 CSV")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--recall", type=float, required=True, help="该阈值下已知 AI 检出率,写进判据")
    args = ap.parse_args()

    scores = {}
    for r in csv.DictReader(open(args.scores, encoding="utf-8")):
        # audio_id 形如 pixabay_quarantine_px_123 → 文件 stem px_123
        stem = r["audio_id"].split("_", 2)[-1]
        scores[stem] = float(r["score"])

    idx = Path(args.index)
    shutil.copy(idx, idx.with_suffix(".csv.bak_before_B"))
    rows = list(csv.DictReader(open(idx, encoding="utf-8")))
    cols = list(rows[0].keys())

    stat = collections.Counter()
    for r in rows:
        if r["corpus"] != args.corpus:
            continue
        stem = Path(r["path"]).stem
        s = scores.get(stem)
        if s is None:
            stat["无分数"] += 1
            continue
        r["notes"] = (r["notes"] + f";mert_score={s:.4f}").strip(";")
        if s < args.threshold:
            r["ai_risk"] = "0"
            r["ai_risk_method"] = (f"MERT 打分 {s:.3f} < 阈值 {args.threshold:.3f}(留出人类 p95)"
                                   f";尺子对已知 AI 检出率仅 {args.recall:.0%},漏检仍可能在")
            r["layer"] = "core" if r["instrumental"] == "True" else "aux"
            stat["→ " + r["layer"]] += 1
        else:
            r["ai_risk"] = "unknown"
            r["ai_risk_method"] = f"MERT 打分 {s:.3f} ≥ 阈值 {args.threshold:.3f},留 quarantine"
            r["layer"] = "quarantine"
            stat["留 quarantine"] += 1

    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"[{args.corpus}] {dict(stat)}")
    hc = collections.Counter(r["layer"] for r in rows if r["side"] == "human")
    print(f"人类侧现在: {dict(hc)}")
    print(f"备份 → {idx.with_suffix('.csv.bak_before_B').name}")


if __name__ == "__main__":
    main()
