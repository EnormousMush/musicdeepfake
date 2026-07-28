"""
SONICS 选样 + 切段:从 10 个 zip 里抽 每版本 1000 首,切中段 30s(mp3 流拷贝),
产出待人声分离的 ship-pack。

流程定位:本脚本(Mac,CPU)→ demucs 伴奏分离(GPU pod)→ crossgen_prep 共同规格。
选样 seed 0 可复现;断点续跑(已存在的输出跳过)。

  python sonics_prep.py --sonics-dir "/Volumes/Seagate /sonics" \
      --out "/Volumes/Seagate /sonics/sel30s" --per-version 1000
"""
import argparse
import csv
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np

ALGO_SLUG = {
    "chirp-v2-xxl-alpha": "sunov2",
    "chirp-v3": "sunov3",
    "chirp-v3.5": "sunov35",
    "udio-30s": "udio30",
    "udio-120s": "udio120",
}


def build_zip_index(sonics_dir):
    """扫描全部 part zip 的目录区:成员名(不含扩展名)-> (zip路径, 成员路径)。"""
    index = {}
    for zp in sorted(sonics_dir.glob("part_*.zip")):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                index[Path(name).stem] = (zp, name)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sonics-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-version", type=int, default=1000)
    ap.add_argument("--seg", type=float, default=30.0, help="切段长度(秒)")
    ap.add_argument("--limit", type=int, default=None, help="干跑:每版本只处理 N 首")
    args = ap.parse_args()

    sonics_dir = Path(args.sonics_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"

    rows = list(csv.DictReader(open(sonics_dir / "fake_songs.csv")))
    rng = np.random.default_rng(0)
    by_algo = {}
    for r in rows:
        by_algo.setdefault(r["algorithm"], []).append(r)

    plan = []
    for algo in sorted(by_algo):
        pool = by_algo[algo]
        idx = rng.permutation(len(pool))[: args.per_version if not args.limit else args.limit]
        for i in idx:
            r = pool[int(i)]
            plan.append(dict(audio_id=f"{ALGO_SLUG[algo]}_{r['filename']}", algo=algo,
                             filename=r["filename"], duration=float(r["duration"]),
                             skip=float(r["skip_time"] or 0)))

    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
    todo = [p for p in plan if p["audio_id"] not in done]
    print(f"计划 {len(plan)} | 已完成 {len(plan)-len(todo)} | 待处理 {len(todo)}", flush=True)
    if not todo:
        return

    print("扫描 zip 目录区……", flush=True)
    index = build_zip_index(sonics_dir)
    print(f"索引 {len(index)} 个成员", flush=True)

    new_manifest = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.writer(mf)
    if new_manifest:
        w.writerow(["audio_id", "algo", "filename", "duration", "start_s", "rel_path"])
    ok, fail, t0 = 0, 0, time.time()
    tmp = out / "_tmp"; tmp.mkdir(exist_ok=True)
    for i, p in enumerate(todo, 1):
        try:
            zp, member = index[p["filename"]]
            ext = Path(member).suffix or ".mp3"
            start = max(p["skip"], (p["duration"] - args.seg) / 2)
            algo_dir = out / ALGO_SLUG[p["algo"]]; algo_dir.mkdir(exist_ok=True)
            dst = algo_dir / f"{p['audio_id']}{ext}"
            if not dst.exists():
                raw = tmp / f"raw{ext}"
                with zipfile.ZipFile(zp) as zf, open(raw, "wb") as f:
                    f.write(zf.read(member))
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                     "-ss", f"{start:.2f}", "-t", f"{args.seg:.2f}",
                     "-i", str(raw), "-c", "copy", str(dst)],
                    check=True, stdin=subprocess.DEVNULL)
            w.writerow([p["audio_id"], p["algo"], p["filename"], p["duration"],
                        round(start, 2), f"{ALGO_SLUG[p['algo']]}/{dst.name}"])
            mf.flush(); ok += 1
        except Exception as e:
            print(f"  ❌ {p['audio_id']}: {e!r}", flush=True); fail += 1
        if i % 200 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            print(f"  {i}/{len(todo)} ({rate:.2f}s/首, 失败 {fail})", flush=True)
    mf.close()
    print(f"\n完成:成功 {ok},失败 {fail} -> {out}")


if __name__ == "__main__":
    main()
