"""
对称 demucs 协议 · 真实侧与域内参照切段:fma-test 450 + suno-test 450 各取原曲中段 30s。

动机:SONICS 假歌过了 demucs,对照的 FMA 也必须过同一把刀(处理历史对称),
否则"demucs 痕迹"本身成为区分真假的新捷径;suno-test 同样过刀 = 域内参照行,
量出 demucs 对域内信号的损耗。切段参数与 sonics_prep 完全一致(30s 中段,流拷贝)。

  python devocal_prep.py --export-manifest "/Volumes/Seagate /frank-suno-round1/crossgen_export/manifest.csv" \
      --fma-dir "/Volumes/Seagate /fma/fma_large" --suno-dir "/Volumes/Seagate /suno_audio" \
      --out "/Volumes/Seagate /sonics/ref30s"
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path


def fma_src(fma_dir, audio_id):
    n = audio_id.split("_")[1]
    p6 = n.zfill(6)
    return fma_dir / p6[:3] / f"{p6}.mp3"


def suno_src(suno_dir, audio_id):
    stem = audio_id[len("suno_"):]
    hits = list(suno_dir.glob(f"*/{stem}.mp3"))
    return hits[0] if hits else None


def dur_of(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-manifest", required=True)
    ap.add_argument("--fma-dir", required=True)
    ap.add_argument("--suno-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    fma_dir, suno_dir = Path(args.fma_dir), Path(args.suno_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"

    rows = list(csv.DictReader(open(args.export_manifest)))
    plan = []
    for r in rows:
        if r["split"] != "test" or r["source"] not in ("fma", "suno"):
            continue
        plan.append(dict(audio_id=r["audio_id"], source=r["source"]))
    if args.limit:
        fma = [p for p in plan if p["source"] == "fma"][: args.limit]
        suno = [p for p in plan if p["source"] == "suno"][: args.limit]
        plan = fma + suno
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
    todo = [p for p in plan if p["audio_id"] not in done]
    print(f"计划 {len(plan)} | 已完成 {len(plan)-len(todo)} | 待处理 {len(todo)}", flush=True)
    if not todo:
        return

    new_manifest = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.writer(mf)
    if new_manifest:
        w.writerow(["audio_id", "source", "src_path", "start_s", "rel_path"])
    ok, fail, t0 = 0, 0, time.time()
    for i, p in enumerate(todo, 1):
        try:
            src = (fma_src(fma_dir, p["audio_id"]) if p["source"] == "fma"
                   else suno_src(suno_dir, p["audio_id"]))
            if src is None or not src.exists():
                raise FileNotFoundError(f"原曲缺失 {p['audio_id']}")
            dur = dur_of(src)
            start = max(0.0, (dur - args.seg) / 2)
            sub = out / f"{p['source']}_ref"; sub.mkdir(exist_ok=True)
            dst = sub / f"{p['audio_id']}.mp3"
            if not dst.exists():
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                     "-ss", f"{start:.2f}", "-t", f"{args.seg:.2f}",
                     "-i", str(src), "-c", "copy", str(dst)],
                    check=True, stdin=subprocess.DEVNULL)
            w.writerow([p["audio_id"], p["source"], str(src), round(start, 2),
                        f"{p['source']}_ref/{dst.name}"])
            mf.flush(); ok += 1
        except Exception as e:
            print(f"  ❌ {p['audio_id']}: {e!r}", flush=True); fail += 1
        if i % 100 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} ({(time.time()-t0)/i:.2f}s/首, 失败 {fail})", flush=True)
    mf.close()
    print(f"\n完成:成功 {ok},失败 {fail} -> {out}")


if __name__ == "__main__":
    main()
