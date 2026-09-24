"""
语料审计取样 + 共同规格切片(2026-09-19)。

给两个验证准备素材:
  A. **器乐验证**(demucs):抽 jamendo_sweep 的曲子,验 `musicinfo.vocalinstrumental`
     这个**元数据字段**到底准不准——我们从没听过任何一个音频文件,
     入库口径 100% 靠这个字段,它准不准决定这 2 万首能不能按现在的说法用。
  B. **AI 验证**(鉴伪打分):同一批目标曲 + 已知人类参照 + 已知 AI 参照,
     三组一起切,保证前处理完全一致(否则分数差可能是前处理造成的)。

共同规格:中段 10s / 16kHz / mono / -23 LUFS(见 vault [[数据初硬性统一]])。
只取中段 10s,不是整曲——实测整曲解码 0.74 s/首,中段切片 0.02 s/首,差 37 倍。

Usage:
  python part1_extraction/audit_prep.py --out SCRATCH/audit --n-target 100 \
      --n-ref-human 300 --n-ref-ai 300
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000
CLIP_S = 10.0
TARGET_LUFS = -23.0

SWEEP = Path("/Volumes/Seagate /honors_paper/1_corpora_real/jamendo_sweep")
CORPORA = Path("/Volumes/Seagate /honors_paper/1_corpora_real")
AI = Path("/Volumes/Seagate /honors_paper/2_corpora_ai")


def to_common_spec(src: Path):
    """中段 10s → 16k mono → -23 LUFS 近似(RMS 归一)。返回 None 表示这首不可用。"""
    try:
        dur = librosa.get_duration(path=str(src))
        if dur < CLIP_S:
            return None
        y, _ = librosa.load(str(src), sr=SR, mono=True,
                            offset=max(0.0, dur / 2 - CLIP_S / 2), duration=CLIP_S)
    except Exception:
        return None
    if len(y) < int(SR * CLIP_S * 0.9):
        return None
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:                        # 全静音片段,丢弃
        return None
    y = y * (10 ** (TARGET_LUFS / 20) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:                       # 防削顶
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def write_group(items, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, ok = [], 0
    for i, (aid, path, meta) in enumerate(items):
        y = to_common_spec(Path(path))
        if y is None:
            continue
        dest = out_dir / f"{aid}.wav"
        sf.write(dest, y, SR)
        rows.append(dict(audio_id=aid, group=tag, src=str(path),
                         clip=str(dest), **meta))
        ok += 1
    print(f"  [{tag}] 切出 {ok}/{len(items)} 片 → {out_dir}")
    return rows


def sample_dir(d: Path, n, rng, exts=(".mp3", ".wav", ".flac")):
    files = [p for p in d.rglob("*") if p.suffix.lower() in exts]
    rng.shuffle(files)
    return files[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-target", type=int, default=100, help="待验曲目(每层)")
    ap.add_argument("--n-ref-human", type=int, default=300)
    ap.add_argument("--n-ref-ai", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    all_rows = []

    # --- 待验:jamendo_sweep 两层,按 manifest 抽,带上元数据好做对照
    for man, sub, tag in [("manifest.csv", "audio", "sweep_main"),
                          ("manifest_audio_post2023.csv", "audio_post2023", "sweep_post2023")]:
        rows = list(csv.DictReader(open(SWEEP / man, encoding="utf-8")))
        rng.shuffle(rows)
        items = [(f"{tag}_{r['track_id']}", r["audio_path"],
                  dict(year=r["year"], artist_id=r["artist_id"],
                       claimed=r["vocalinstrumental"], ai_risk=r.get("ai_risk", "")))
                 for r in rows[:args.n_target]]
        all_rows += write_group(items, out / tag, tag)

    # --- 人类参照:jamendo2025 器乐白名单(已知口径) + FMA
    jam_inst = Path("part3_detector/diagnostics/jamendo_instrumental.txt")
    if jam_inst.exists():
        names = [l.strip() for l in open(jam_inst) if l.strip()]
        rng.shuffle(names)
        j = CORPORA / "jamendo2025"
        items = []
        for nm in names[: args.n_ref_human // 2]:
            hits = list(j.rglob(f"{nm}.mp3"))
            if hits:
                items.append((f"ref_jam_{nm}", hits[0], dict(claimed="instrumental")))
        all_rows += write_group(items, out / "ref_human_jam", "ref_human")

    fma = CORPORA / "fma" / "fma_large"
    if fma.exists():
        items = [(f"ref_fma_{p.stem}", p, dict(claimed="unknown"))
                 for p in sample_dir(fma, args.n_ref_human // 2, rng)]
        all_rows += write_group(items, out / "ref_human_fma", "ref_human")

    # --- AI 参照:自产六家 + suno
    for name, d in [("generators", AI / "generators"), ("suno", AI / "suno_audio")]:
        if not d.exists():
            continue
        items = [(f"ref_ai_{name}_{p.stem}", p, dict(claimed="ai"))
                 for p in sample_dir(d, args.n_ref_ai // 2, rng)]
        all_rows += write_group(items, out / f"ref_ai_{name}", "ref_ai")

    idx = out / "index.csv"
    cols = sorted({k for r in all_rows for k in r})
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    import collections
    print(f"\n总计 {len(all_rows)} 片 → {idx}")
    print("分组:", dict(collections.Counter(r["group"] for r in all_rows)))


if __name__ == "__main__":
    main()
