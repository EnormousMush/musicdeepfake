"""
带宽匹配消融 — 把两类都低通到同一截止频率,重抽特征、逐层重探针,和原始逐层 EER 并排对比。

回答:近 0% 的分离里,有多少是"Suno 更亮/高频更多"(年代/制作代差)喂出来的?
  - 低通后 EER 大幅上升  -> 亮度/带宽差是主要驱动(年代混淆主要在高频)
  - 基本不动(仍 ~0-2%) -> 分离是宽带的,得靠重建法去混淆

对两类做完全相同的滤波(Butterworth 10 阶,零相位 filtfilt)—— 不改采样率,只削内容,
避免引入新的重采样差异。特征缓存到 features_lp{cutoff}/,重跑秒级恢复。

Usage (server, venv active, GPU1; MERT 6000 clips 约 1-2h):
  python diagnostics/bandwidth_ablation.py --data-dir data_store/subset_export_round1 \
      --encoder mert --cutoff 8000
  加 --limit 20 先干跑验证能动。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer


def probe_all_layers(F, y, sp):
    """[N,L,2D] -> list of (layer, val_eer, test_eer)."""
    tr, va, te = sp == "train", sp == "val", sp == "test"
    out = []
    for L in range(F.shape[1]):
        clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
        ev = compute_eer(y[va], linear_clf.score(clf, F[va, L]))["eer"] if va.sum() else float("nan")
        et = compute_eer(y[te], linear_clf.score(clf, F[te, L]))["eer"] if te.sum() else float("nan")
        out.append((L, ev, et))
    return out


def load_cached(rows, cache):
    F, y, sp = [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); y.append(int(r["label"])); sp.append(r["split"])
    return (np.stack(F), np.array(y), np.array(sp)) if F else (None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--cutoff", type=int, default=8000, help="低通截止 Hz(两类一视同仁)")
    ap.add_argument("--limit", type=int, default=None, help="每类只跑前 N 个(干跑用)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from scipy.signal import butter, sosfiltfilt
    from encoders.ssl import SSLEncoder

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    if args.limit:
        keep, seen = [], {0: 0, 1: 0}
        for r in rows:
            lb = int(r["label"])
            if seen[lb] < args.limit:
                keep.append(r); seen[lb] += 1
        rows = keep
        print(f"Dry-run: {len(rows)} clips")

    enc = SSLEncoder(args.encoder, device=args.device)
    print(f"Encoder: {args.encoder} on {enc.device} | lowpass cutoff = {args.cutoff} Hz")

    lp_cache = data_dir / f"features_lp{args.cutoff}" / args.encoder
    lp_cache.mkdir(parents=True, exist_ok=True)

    sos_cache = {}
    F, y, sp, failures = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cpath = lp_cache / f"{r['audio_id']}.npy"
        try:
            if cpath.exists():
                f = np.load(cpath)
            else:
                wav, srate = sf.read(data_dir / r["rel_path"])
                wav = np.asarray(wav, dtype=np.float64)
                if srate not in sos_cache:
                    sos_cache[srate] = butter(10, args.cutoff, btype="low", fs=srate, output="sos")
                wav = sosfiltfilt(sos_cache[srate], wav)
                f = enc.encode_all_layers(np.asarray(wav, dtype=np.float32), srate)
                np.save(cpath, f)
            F.append(f); y.append(int(r["label"])); sp.append(r["split"])
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 100 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)")

    F = np.stack(F); y = np.array(y); sp = np.array(sp)
    lp = probe_all_layers(F, y, sp)

    orig_cache = data_dir / "features" / args.encoder
    Fo, yo, spo = load_cached(rows, orig_cache)
    orig = probe_all_layers(Fo, yo, spo) if Fo is not None else None

    print(f"\n=== 逐层 EER:原始(全带宽) vs 低通 {args.cutoff} Hz ===")
    print(f"{'layer':>5} {'orig val':>9} {'orig test':>10} {'lp val':>9} {'lp test':>10}")
    for L, ev, et in lp:
        if orig:
            _, oev, oet = orig[L]
            print(f"{L:>5} {oev*100:>8.2f}% {oet*100:>9.2f}% {ev*100:>8.2f}% {et*100:>9.2f}%")
        else:
            print(f"{L:>5} {'--':>9} {'--':>10} {ev*100:>8.2f}% {et*100:>9.2f}%")

    best = min(lp, key=lambda p: p[1] if np.isfinite(p[1]) else 9)
    print(f"\n低通后最佳层(val):layer {best[0]}  val {best[1]*100:.2f}%  test {best[2]*100:.2f}%")
    print("判读:lp test 相比 orig test 大幅上升 -> 高频/亮度差是主要驱动;基本不动 -> 分离是宽带的。")
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")


if __name__ == "__main__":
    main()
