"""
fma30 平铺伪迹修复导出(2026-08-17,代码审计发现 #1 的处置)。

病灶:round1 用 offset 10s + crop 30s 打在 fma_large 的 ~30s 摘录上,_fix_length
把开头 ~10s 精确复制到尾部(corr=1.0000),伪迹只在 fma 侧、与类别相关;
h2_export 的 fma30 又整片复用了这批 FLAC。

修复:从 fma_large **原盘**重导——offset 0,crop = min(30s, 源时长 − 0.02s),
**绝不平铺**(宁可 29.98s 也不造假样本)。输出到新窗口 fma30fix(旧 fma30 留作法证),
manifest 追加 source=fma30fix 行(split/label 原样)。

Usage(Mac,Seagate 挂载,约 20 分钟):
  python3 part3_detector/diagnostics/h2_fma30_refix.py \
    --out "/Volumes/Seagate /honors_paper/3_workpacks/frank-suno-round1/h2_export"
  加 --limit 5 干跑。之后 extract_midwindow.py --sources fma30fix 重提特征。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import librosa
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from part2_analysis.preprocess.audio import load_canonical

FMA_LARGE = "/Volumes/Seagate /honors_paper/1_corpora_real/fma/fma_large"   # round1 旧根已重组至此
SPEC = dict(sr=16000, mono=True, loudness_lufs=-23.0)


def src_path(audio_id: str) -> Path:
    tid = audio_id[len("fma_"):].zfill(6)
    return Path(FMA_LARGE) / tid[:3] / f"{tid}.mp3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="h2_export 目录(含 manifest.csv)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    man = out / "manifest.csv"
    rows = [r for r in csv.DictReader(open(man)) if r["source"] == "fma30"]
    if args.limit:
        rows = rows[: args.limit]
    done = {r["audio_id"] for r in csv.DictReader(open(man)) if r["source"] == "fma30fix"}
    (out / "audio" / "fma30fix").mkdir(parents=True, exist_ok=True)
    mf = open(man, "a", newline="")
    w = csv.DictWriter(mf, fieldnames=["audio_id", "source", "label", "split", "rel_path"])
    n_ok, n_err, t0 = 0, 0, time.time()
    for r in rows:
        aid = r["audio_id"]
        if aid in done:
            continue
        src = src_path(aid)
        dst = out / "audio" / "fma30fix" / f"{aid}.flac"
        try:
            dur = librosa.get_duration(path=str(src))
            crop = min(30.0, max(1.0, dur - 0.02))   # 绝不平铺
            y = load_canonical(str(src), dict(SPEC, offset_s=0.0, crop_s=crop))
            sf.write(dst, y, SPEC["sr"])
        except Exception as e:
            n_err += 1
            print(f"  {aid}: ERROR {repr(e)[:70]}", flush=True)
            continue
        w.writerow(dict(audio_id=aid, source="fma30fix", label=r["label"],
                        split=r["split"], rel_path=f"audio/fma30fix/{aid}.flac"))
        mf.flush()
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"  {n_ok} exported ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {n_ok} exported, {n_err} errors", flush=True)


if __name__ == "__main__":
    main()
