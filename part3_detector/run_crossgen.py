"""
跨生成器测试 · 服务器端 — Suno-训练的探针,能不能抓住别的生成器?

吃 crossgen_prep.py 打包的共同规格文件夹(10s/16k/LUFS,含 suno/fma/各生成器):
  1) 抽 MERT 全层特征(缓存);
  2) 用 suno-vs-fma 的 train split 训逐层 probe(域内基线,应仍近 0%);
  3) 按 val 选最佳层 L*;
  4) 每个外部生成器 g:EER( g 的全部 clips ∪ fma 的 test clips ),在 L* 和逐层最优两档报告。

判读(预先写死):
  - 生成器 EER 仍低(接近域内)  -> 探针抓到了跨生成器的通用信号 -> "生成指纹"假设加分;
  - 生成器 EER 大幅劣化(趋 50%)-> Suno-专属(或制作纹理)-> 与文献一致,"数据集检测"警报。
注意:FakeMusicCaps 内容分布(MusicCaps caption,含类人声)≠ 我们的 instrumental,
结论要连同这个 content shift 一起说,不能只报数。

Usage (server, venv active, GPU1;11k×10s 抽特征约 1h):
  python run_crossgen.py --data-dir data_store/crossgen_export --encoder mert
  加 --limit 20 干跑。
"""
import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from encoders.ssl import SSLEncoder
    enc = SSLEncoder(args.encoder, device=args.device)
    print(f"Encoder: {args.encoder} on {enc.device}")

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    if args.limit:
        keep, seen = [], {}
        for r in rows:
            if seen.get(r["source"], 0) < args.limit:
                keep.append(r); seen[r["source"]] = seen.get(r["source"], 0) + 1
        rows = keep
        print(f"Dry-run: {len(rows)} clips")

    cache = data_dir / "features" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)

    F, y, sp, src, failures = [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cpath = cache / f"{r['audio_id']}.npy"
        try:
            if cpath.exists():
                f = np.load(cpath)
            else:
                wav, srate = sf.read(data_dir / r["rel_path"])
                f = enc.encode_all_layers(np.asarray(wav, dtype=np.float32), srate)
                np.save(cpath, f)
            F.append(f); y.append(int(r["label"])); sp.append(r["split"]); src.append(r["source"])
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 200 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)", flush=True)

    F = np.stack(F); y = np.array(y); sp = np.array(sp); src = np.array(src)
    n_layers = F.shape[1]
    in_domain = (src == "suno") | (src == "fma")
    gens = sorted(set(src) - {"suno", "fma"})
    print(f"\nclips: {len(y)}  | 域内 {int(in_domain.sum())} | 生成器 {gens}")

    tr = in_domain & (sp == "train")
    va = in_domain & (sp == "val")
    te = in_domain & (sp == "test")
    fma_te = (src == "fma") & (sp == "test")

    def safe_eer(mask, L, clf):
        """空 split / 单类时返回 nan(干跑 --limit 时会发生),不崩。"""
        if mask.sum() == 0 or len(set(y[mask].tolist())) < 2:
            return float("nan")
        return compute_eer(y[mask], linear_clf.score(clf, F[mask, L]))["eer"]

    # ---- 逐层:域内基线 + 各生成器 ----
    clfs, base = [], []
    print(f"\n=== 域内基线(suno vs fma @ 共同规格 10s/16k)===", flush=True)
    print(f"{'layer':>5} {'val':>8} {'test':>8}")
    for L in range(n_layers):
        clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
        clfs.append(clf)
        ev = safe_eer(va, L, clf)
        et = safe_eer(te, L, clf)
        base.append((L, ev, et))
        print(f"{L:>5} {ev*100:>7.2f}% {et*100:>7.2f}%", flush=True)
    valid = [p for p in base if np.isfinite(p[1])]
    Lstar = min(valid, key=lambda p: p[1])[0] if valid else 0
    print(f"最佳层(val):L* = {Lstar}  (val {base[Lstar][1]*100:.2f}%, test {base[Lstar][2]*100:.2f}%)")

    def gen_eer(g, L):
        m = (src == g) | fma_te
        if m.sum() == 0 or len(set(y[m].tolist())) < 2:
            return float("nan")
        return compute_eer(y[m], linear_clf.score(clfs[L], F[m, L]))["eer"]

    print(f"\n=== 跨生成器 EER 矩阵(每个生成器 vs fma-test)===")
    print(f"{'generator':>24} {'@L*='+str(Lstar):>9} {'best-layer(EER@L)':>20}")
    for g in gens:
        e_star = gen_eer(g, Lstar)
        per_l = [(L, e) for L in range(n_layers) for e in [gen_eer(g, L)] if np.isfinite(e)]
        Lb, eb = min(per_l, key=lambda p: p[1]) if per_l else (Lstar, float("nan"))
        print(f"{g:>24} {e_star*100:>8.2f}% {eb*100:>13.2f}% (L{Lb})", flush=True)
    print(f"{'(域内 suno, 参照)':>24} {base[Lstar][2]*100:>8.2f}%")

    print("\n判读:生成器 EER ≈ 域内 -> 通用信号;趋 50% -> Suno-专属/数据集检测。"
          "\n(记得连同 content shift —— MusicCaps caption vs instrumental —— 一起解读)")
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")


if __name__ == "__main__":
    main()
