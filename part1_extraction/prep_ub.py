"""
U-B 训练前处理:把 core 全量统一成 16 kHz mono、定长窗(2026-09-23)。

为什么必须做(不是可选):
  SpecTTTra 的 AudioDataset 用 librosa.load(sr=None) —— **不重采样**,按原生采样率
  裁 max_time×16000 个样本再喂给假定 16 kHz 的梅尔谱。我们的 core 混着 48k flac、
  48k mp3、44.1k mp3、16k wav:48k 文件裁出来的是 40 s 真音频被当 120 s 用。
  模型会直接学采样率。SONICS 自己的数据全是 16k mp3,所以他们没踩到。

  另:fakemusiccaps 是 10 s 片段,按 120 s 窗会被零填充 92% —— "AI = 大段静音"捷径。
  所以窗长必须统一,且两侧一样。

输出:<out>/<split>/<uid>.wav(16k mono int16),每首取 --n-windows 个窗(随机起点,
短于窗的整首零填充到窗长——两侧同规则),并生成 SpecTTTra 要的 train/valid/test.csv。

Usage:
  python part1_extraction/prep_ub.py --splits _INDEX/splits_UB --out /Volumes/.../3_workpacks/ub_16k \
      --window 10 --n-windows 1 --workers 8
"""
from __future__ import annotations
import argparse, csv, concurrent.futures as cf, random
from pathlib import Path
import librosa, numpy as np, soundfile as sf

SR = 16000

def one(job):
    src, dst_base, window, n_win, seed = job
    try:
        y, _ = librosa.load(src, sr=SR, mono=True)
    except Exception as e:
        return src, [], f"decode {type(e).__name__}"
    L = int(window * SR); rng = random.Random(seed); outs = []
    if len(y) < int(0.5 * L):
        return src, [], "too_short"
    for k in range(n_win):
        if len(y) <= L:
            seg = np.zeros(L, np.float32); seg[: len(y)] = y
        else:
            s = rng.randint(0, len(y) - L); seg = y[s : s + L]
        p = Path(f"{dst_base}_w{k}.wav"); p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, seg, SR, subtype="PCM_16"); outs.append(str(p))
    return src, outs, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, required=True, help="秒")
    ap.add_argument("--n-windows", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="烟测:每 split 只处理前 N 首")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        rows = list(csv.DictReader(open(Path(a.splits) / f"{split}.csv", encoding="utf-8")))
        if a.limit: rows = rows[: a.limit]
        jobs = [(r["filepath"], str(out / split / f"{r['corpus']}__{Path(r['filepath']).stem}"),
                 a.window, a.n_windows, i) for i, r in enumerate(rows)]
        ok = short = fail = 0
        with open(out / f"{split}.csv", "w", newline="", encoding="utf-8") as f, \
             cf.ProcessPoolExecutor(a.workers) as ex:
            w = csv.writer(f); w.writerow(["filepath", "target", "corpus", "family", "src"])
            for (src, outs, err), r in zip(ex.map(one, jobs, chunksize=16), rows):
                if err == "too_short": short += 1; continue
                if err: fail += 1; continue
                for o in outs: w.writerow([o, r["target"], r["corpus"], r["family"], src])
                ok += 1
        print(f"{split:5s} ok {ok}  too_short {short}  fail {fail}", flush=True)

if __name__ == "__main__":
    main()
