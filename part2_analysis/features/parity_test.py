"""新旧对拍(2026-08-20 大修验收):同一批 clip,旧六模块 vs 新 features 包,
全设计列数值必须一致(浮点容忍 1e-9;事故列 ibis_*/duration/hop_length 豁免 —— 新版故意不产)。

Usage(Mac,Seagate 挂载):
  python part2_analysis/features/parity_test.py \
      --data-dir "/Volumes/Seagate /honors_paper/3_workpacks/frank-suno-round1/crossgen_export" \
      --sources suno,fma --n 10
  加 --set h2_mid --data-dir <h2_export> --sources suno30,fma30,jam30 对拍中窗集。
"""
import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "part2_analysis"))  # 旧模块 `from analysis_utils import` 所需

from part2_analysis.features.extract import one_clip as new_one_clip

EXEMPT_PREFIX = ("ibis_",)
EXEMPT = {"duration", "hop_length"}


def old_h1(path, meta):
    """复刻旧 extract_features.one_clip(含旧 _flatten / 旧 SKIP_KEYS)。"""
    from part2_analysis.spectral import analyze_spectral
    from part2_analysis.timbral import analyze_timbral
    from part2_analysis.dynamics import analyze_dynamics
    from part2_analysis.rhythm import analyze_rhythm
    from part2_analysis.quantize_deg import analyze_quantization
    from part2_analysis.key import estimate_key
    from part2_analysis.extract_features import _flatten
    row = dict(meta)
    for name, fn in [("spec", analyze_spectral), ("timb", analyze_timbral),
                     ("dyn", analyze_dynamics), ("rhy", analyze_rhythm),
                     ("qnt", analyze_quantization)]:
        try:
            d = fn(path)
            if "error" in d:
                row[f"{name}_error"] = d["error"]
            else:
                _flatten({k: v for k, v in d.items() if not k.startswith("_")}, row)
        except Exception as e:
            row[f"{name}_error"] = repr(e)[:80]
    try:
        bk, bc, ak, ac = estimate_key(path)
        row["best_key"] = str(bk); row["best_corr"] = float(bc)
        if ak is not None:
            row["alt_key"] = str(ak); row["alt_corr"] = float(ac)
    except Exception as e:
        row["key_error"] = repr(e)[:80]
    return row


def old_h2mid(path, meta):
    from part2_analysis.midwindow import analyze_midwindow
    row = dict(meta)
    row.update(analyze_midwindow(path))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="h1_10s", choices=["h1_10s", "h2_mid"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(Path(args.data_dir) / "manifest.csv"))
            if r["source"] in set(args.sources.split(","))]
    rng = random.Random(20260820)
    rng.shuffle(rows)
    picks = rows[: args.n]
    print(f"parity {args.set}: {len(picks)} clips", flush=True)

    n_pass, worst = 0, (0.0, "", "")
    for r in picks:
        path = str(Path(args.data_dir) / r["rel_path"])
        meta = {k: r.get(k, "") for k in ("audio_id", "source", "split", "label")}
        old = old_h1(path, meta) if args.set == "h1_10s" else old_h2mid(path, meta)
        new = new_one_clip((path, meta, args.set))
        old_f = {k: v for k, v in old.items()
                 if k not in meta and k not in EXEMPT and not k.startswith(EXEMPT_PREFIX)}
        bad = []
        for k, v in old_f.items():
            if k not in new:
                bad.append(f"missing:{k}")
            elif isinstance(v, float) or isinstance(new.get(k), float):
                diff = abs(float(v) - float(new[k]))
                if diff > 1e-9:
                    bad.append(f"{k}: old={v} new={new[k]}")
                if diff > worst[0]:
                    worst = (diff, k, r["audio_id"])
            elif str(v) != str(new.get(k)):
                bad.append(f"{k}: old={v!r} new={new[k]!r}")
        extra = [k for k in new if k not in old and k not in meta and not k.endswith("_error")]
        if extra:
            bad.append(f"extra_cols:{extra}")
        if bad:
            print(f"  ✗ {r['audio_id']} ({len(bad)} diffs): {bad[:6]}", flush=True)
        else:
            n_pass += 1
            print(f"  ✓ {r['audio_id']} ({len(old_f)} cols equal)", flush=True)
    print(f"\nPARITY {'PASS' if n_pass == len(picks) else 'FAIL'}: "
          f"{n_pass}/{len(picks)} clips, worst diff {worst[0]:.2e} ({worst[1]})", flush=True)


if __name__ == "__main__":
    main()
