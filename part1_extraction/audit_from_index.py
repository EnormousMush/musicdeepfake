"""
按索引做审计:完整性全量 + 分层抽样(2026-09-21)。

前面的审计脚本都是按目录写死的;定稿索引建好之后,一切都应该从
`final_index.csv` 出发——这样"测了什么"和"入库了什么"永远对得上。

两个子命令:
  verify  对指定 layer 的**全部**文件做完整性快检(魔数 + 大小),
          并统计各语料的 codec / inst_tier / ai_risk 分布
  sample  按 (corpus, layer) 分层抽样,切共同规格片段,供 demucs / MERT 审计

抽样的重点:**`tag` 档判据从未被音频验证过**。
FMA 的 14,755 首靠"上传者打的 Instrumental 流派标签"进的 core,
MTAT 的 1,158 首靠"no-vocals 众包标签"——两者都没听过一首。
`field` 档(Jamendo)已经测过,明确人声 ~4%。

Usage:
  .venv_audit/bin/python part1_extraction/audit_from_index.py verify \
      --index final_index.csv --side human --layer core
  .venv_audit/bin/python part1_extraction/audit_from_index.py sample \
      --index final_index.csv --out DIR --spec "fma:core:120,magnatagatune:core:120"
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR, CLIP_S, TARGET_LUFS = 16000, 10.0, -23.0
_MAGIC = {
    "mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"),
    "wav": (b"RIF",), "flac": (b"fLa",), "ogg": (b"Ogg",),
    "m4a": (b"\x00\x00\x00",), "au": (b".sn",),
}


def check(p: Path, codec: str):
    if not p.exists():
        return "文件不存在"
    if p.stat().st_size < 10_000:
        return f"过小({p.stat().st_size}B)"
    magic = _MAGIC.get(codec)
    if magic:
        with open(p, "rb") as f:
            head = f.read(4)
        if not any(head[: len(m)] == m for m in magic):
            return "魔数不符"
    return ""


def load_rows(index, side=None, layer=None, corpus=None):
    rows = []
    with open(index, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if side and r["side"] != side:
                continue
            if layer and r["layer"] != layer:
                continue
            if corpus and r["corpus"] != corpus:
                continue
            rows.append(r)
    return rows


def cmd_verify(args):
    rows = load_rows(args.index, args.side, args.layer)
    print(f"[verify] {args.side or '全部'} / {args.layer or '全部层'}:{len(rows)} 首\n")
    bad = collections.Counter()
    bad_ex = collections.defaultdict(list)
    for i, r in enumerate(rows, 1):
        why = check(Path(r["path"]), r["native_codec"])
        if why:
            bad[f"{r['corpus']}:{why}"] += 1
            if len(bad_ex[r["corpus"]]) < 2:
                bad_ex[r["corpus"]].append(r["path"])
        if i % 20000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\n完整性:通过 {len(rows)-sum(bad.values())}/{len(rows)}")
    if bad:
        for k, v in bad.most_common():
            print(f"   {k:44s} {v}")
        for c, ex in bad_ex.items():
            print(f"   例 {c}: {ex[0]}")
    else:
        print("   ✅ 全部通过")

    print("\n构成:")
    for key in ("corpus", "native_codec", "inst_tier", "ai_risk", "proc_history"):
        c = collections.Counter(r.get(key, "") for r in rows)
        print(f"   {key:14s} {dict(c.most_common(8))}")


def to_clip(src: Path):
    import librosa
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
    if rms < 1e-6:
        return None
    y = y * (10 ** (TARGET_LUFS / 20) / rms)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y *= 0.99 / pk
    return y.astype(np.float32)


def cmd_sample(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    index_rows = []
    for spec in args.spec.split(","):
        corpus, layer, n = spec.split(":")
        pool = load_rows(args.index, None, layer, corpus)
        rng.shuffle(pool)
        tag = f"{corpus}_{layer}"
        d = out / tag
        d.mkdir(exist_ok=True)
        ok = 0
        for r in pool:
            if ok >= int(n):
                break
            y = to_clip(Path(r["path"]))
            if y is None:
                continue
            aid = f"{tag}_{Path(r['path']).stem}"
            sf.write(d / f"{aid}.wav", y, SR)
            index_rows.append(dict(audio_id=aid, group=tag, src=r["path"],
                                   clip=str(d / f"{aid}.wav"),
                                   claimed=r["instrumental"],
                                   inst_tier=r["inst_tier"],
                                   method=r["instrumental_method"][:60],
                                   year="", artist_id="", ai_risk=r["ai_risk"]))
            ok += 1
        print(f"  {tag}: 切出 {ok} 片(池 {len(pool)})")

    idx = out / "index.csv"
    old = list(csv.DictReader(open(idx, encoding="utf-8"))) if idx.exists() else []
    old = [r for r in old if r.get("group") not in {x["group"] for x in index_rows}]
    cols = sorted({k for r in old + index_rows for k in r})
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in old + index_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nindex.csv 共 {len(old)+len(index_rows)} 行 → {idx}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--index", required=True)
    v.add_argument("--side", default="")
    v.add_argument("--layer", default="")
    s = sub.add_parser("sample")
    s.add_argument("--index", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--spec", required=True,
                   help="corpus:layer:n，逗号分隔，如 fma:core:120,fma:aux:120")
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"verify": cmd_verify, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
