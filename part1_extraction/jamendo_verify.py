"""
jamendo_sweep 语料完整性校验(2026-09-19)。

下载是并发 + 可中断的,所以必须独立验一遍,不能信 manifest 自己的说法。查四件事:
  1. **manifest ↔ 磁盘双向一致**:manifest 里有但盘上没有(丢文件)、
     盘上有但 manifest 里没有(孤儿文件,中断时写盘成功但 CSV 没落)
  2. **文件级完好**:mp3 魔数 + 大小下限,挑出半截文件
  3. **真能解码**:用 ffprobe 读实际时长,和 API 报的时长对比——
     魔数对但内容截断的文件只有解码才查得出来
  4. **去重与口径**:track_id 是否唯一、是否 100% instrumental

Usage:
  python part1_extraction/jamendo_verify.py --out DIR            # 快检(1+2+4)
  python part1_extraction/jamendo_verify.py --out DIR --probe    # 加解码抽检
  python part1_extraction/jamendo_verify.py --out DIR --probe --probe-all
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")
MIN_BYTES = 10_000

LAYERS = [("manifest.csv", "audio", "主批 (2004–2022 零风险)"),
          ("manifest_audio_post2023.csv", "audio_post2023", "现代层 (2023–2026)")]


def check_file(p: Path):
    """返回 (ok, 原因)。只做不解码的快检。"""
    if not p.exists():
        return False, "文件不存在"
    sz = p.stat().st_size
    if sz < MIN_BYTES:
        return False, f"过小({sz}B)"
    with open(p, "rb") as f:
        head = f.read(3)
    if not any(head[: len(m)] == m for m in _MP3_MAGIC):
        return False, "魔数不对"
    return True, ""


def probe_duration(p: Path):
    """用 ffprobe 读实际时长;解码不了就是坏文件。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(p)],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        return float(r.stdout.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="jamendo_sweep 完整性校验")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe", action="store_true", help="用 ffprobe 验解码")
    ap.add_argument("--probe-all", action="store_true", help="全量解码(默认抽 400 首)")
    ap.add_argument("--probe-n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tol", type=float, default=3.0, help="时长容差(秒)")
    args = ap.parse_args()

    out = Path(args.out)
    has_ffprobe = shutil.which("ffprobe") is not None
    if args.probe and not has_ffprobe:
        print("⚠️  找不到 ffprobe,跳过解码检查(brew install ffmpeg)\n")

    report = {}
    grand_ok = True

    for man, sub, label in LAYERS:
        mp, ad = out / man, out / sub
        if not mp.exists():
            continue
        rows = list(csv.DictReader(open(mp, encoding="utf-8")))
        disk = {p.name: p for p in ad.glob("*.mp3")}

        print(f"{'='*66}\n■ {label}\n{'='*66}")
        print(f"manifest {len(rows)} 行   磁盘 {len(disk)} 个文件")

        # --- 1. 双向一致
        want = {Path(r["audio_path"]).name for r in rows}
        missing = sorted(want - set(disk))
        orphan = sorted(set(disk) - want)
        print(f"\n[1] 双向一致")
        print(f"    manifest 有而盘上无 : {len(missing)}")
        print(f"    盘上有而 manifest 无: {len(orphan)}  ← 中断留下的孤儿")
        for x in missing[:3]:
            print(f"        缺: {x}")
        for x in orphan[:3]:
            print(f"        孤儿: {x}")

        # --- 2. 文件级快检
        bad = []
        for r in rows:
            p = Path(r["audio_path"])
            ok, why = check_file(p)
            if not ok:
                bad.append((r["track_id"], why))
        cnt = collections.Counter(w for _, w in bad)
        print(f"\n[2] 文件完好 (魔数 + 大小)")
        print(f"    通过 {len(rows)-len(bad)}/{len(rows)}" +
              (f"   问题: {dict(cnt)}" if bad else "   ✅ 全部通过"))

        # --- 3. 口径
        vi = collections.Counter(r["vocalinstrumental"] for r in rows)
        tids = [r["track_id"] for r in rows]
        dup = len(tids) - len(set(tids))
        yrs = collections.Counter(r["year"] for r in rows)
        arts = collections.Counter(r["artist_id"] for r in rows)
        print(f"\n[3] 口径")
        print(f"    器乐      : {vi.get('instrumental',0)}/{len(rows)}"
              + ("  ✅" if vi.get('instrumental', 0) == len(rows) else f"  ⚠️ 其他 {dict(vi)}"))
        print(f"    track_id  : {len(set(tids))} 唯一, 重复 {dup}"
              + ("  ✅" if dup == 0 else "  ⚠️"))
        print(f"    艺术家    : {len(arts)} 位, 单人最多 {max(arts.values())} 首")
        print(f"    年份      : {min(yrs)}–{max(yrs)}")

        # --- 4. 解码抽检
        probe_bad = []
        if args.probe and has_ffprobe:
            good = [r for r in rows if check_file(Path(r["audio_path"]))[0]]
            sample = good if args.probe_all else random.Random(0).sample(
                good, min(args.probe_n, len(good)))
            print(f"\n[4] 解码校验 ({len(sample)} 首"
                  f"{'全量' if args.probe_all else '抽样'})")
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(probe_duration, Path(r["audio_path"])): r
                        for r in sample}
                for fut in cf.as_completed(futs):
                    r = futs[fut]
                    d = fut.result()
                    if d is None:
                        probe_bad.append((r["track_id"], "解码失败"))
                    elif r["duration"]:
                        want_s = float(r["duration"])
                        if abs(d - want_s) > max(args.tol, want_s * 0.05):
                            probe_bad.append(
                                (r["track_id"], f"时长不符 盘上{d:.0f}s vs API{want_s:.0f}s"))
            print(f"    通过 {len(sample)-len(probe_bad)}/{len(sample)}" +
                  ("   ✅ 全部通过" if not probe_bad else ""))
            for t, w in probe_bad[:5]:
                print(f"        {t}: {w}")

        layer_ok = not (missing or bad or probe_bad or dup
                        or vi.get("instrumental", 0) != len(rows))
        grand_ok &= layer_ok
        print(f"\n    → {'✅ 本层通过' if layer_ok else '⚠️ 本层有问题(见上)'}\n")

        report[label] = dict(
            manifest=len(rows), disk=len(disk), missing=len(missing),
            orphan=len(orphan), bad_file=len(bad), dup_track_id=dup,
            instrumental=vi.get("instrumental", 0),
            artists=len(arts), probe_checked=len(probe_bad) and -1 or 0,
            probe_bad=len(probe_bad), ok=layer_ok,
        )

    total = sum(v["manifest"] for v in report.values())
    print(f"{'='*66}")
    print(f"合计入库 {total} 首   {'✅ 全部校验通过' if grand_ok else '⚠️ 存在问题'}")
    print(f"{'='*66}")
    (out / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 → {out/'verify_report.json'}")


if __name__ == "__main__":
    main()
