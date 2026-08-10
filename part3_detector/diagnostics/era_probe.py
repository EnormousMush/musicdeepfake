"""
年代混淆拆旗 · SSL 侧三连测(2026-08-10 立项,配套 jamendo2025 现代人类语料)。

背景:主线一直悬着一面红旗——suno(2025) vs fma(2010s) 的 ~0.1% EER,
到底是"AI 指纹"还是"年代/制作指纹"?jamendo(2024–26,与 fma 同属独立/CC 生态,
唯一变化是年代)进场后,可以三个角度直接开火:

  E1 同代对决:probe 训 suno-train vs jamendo-train,test EER。
     仍近 0% -> SSL 信号不靠年代;大幅劣化 -> 年代红旗坐实。
  E2 旧探针跨年代迁移:probe 训 suno vs fma(原域内配方),
     eval 在 suno-test vs jamendo-test。低 -> "real"定义不锚定年代;高 -> 探针学了年代。
  E3 年代信号本身:probe 训 fma-train vs jamendo-train(人类 vs 人类,label=年代),
     test EER。这是"编码器里年代信号有多强"的直接测量,低 EER = 年代信号强。

特征缓存与 run_crossgen 完全共享(features/<encoder>/<audio_id>.npy),
jamendo 首跑会现算(3k clips,GPU 约 20min),之后秒读。

Usage(服务器,tmux,venv):
  python -u diagnostics/era_probe.py --data-dir data_store/crossgen_export --encoder mert 2>&1 | tee batchE_mert.txt
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer

SOURCES = ("suno", "fma", "jamendo")


def load_features(args):
    data_dir = Path(args.data_dir)
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv"))
            if r["source"] in SOURCES]
    if args.limit:
        keep, seen = [], {}
        for r in rows:
            if seen.get(r["source"], 0) < args.limit:
                keep.append(r); seen[r["source"]] = seen.get(r["source"], 0) + 1
        rows = keep
        print(f"Dry-run: {len(rows)} clips")

    from encoders.ssl import SSLEncoder
    enc = None
    cache = data_dir / "features" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)
    F, sp, src, failures = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cpath = cache / f"{r['audio_id']}.npy"
        try:
            if cpath.exists():
                f = np.load(cpath)
            else:
                if enc is None:                      # 缓存全命中时不加载模型
                    enc = SSLEncoder(args.encoder, device=args.device)
                wav, srate = sf.read(data_dir / r["rel_path"])
                f = enc.encode_all_layers(np.asarray(wav, dtype=np.float32), srate)
                np.save(cpath, f)
            F.append(f); sp.append(r["split"]); src.append(r["source"])
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 200 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)", flush=True)
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")
    return np.stack(F), np.array(sp), np.array(src)


def probe_curve(F, y, tr, va, te, tag):
    """逐层训练+评测,返回 (Lstar, val, test) 并打印曲线。"""
    n_layers = F.shape[1]
    print(f"\n=== {tag} ===", flush=True)
    print(f"{'layer':>5} {'val':>8} {'test':>8}")
    out = []
    for L in range(n_layers):
        clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
        ev = compute_eer(y[va], linear_clf.score(clf, F[va, L]))["eer"] if va.sum() else float("nan")
        et = compute_eer(y[te], linear_clf.score(clf, F[te, L]))["eer"] if te.sum() else float("nan")
        out.append((L, ev, et, clf))
        print(f"{L:>5} {ev*100:>7.2f}% {et*100:>7.2f}%", flush=True)
    valid = [p for p in out if np.isfinite(p[1])]
    Ls, ev, et, _ = min(valid, key=lambda p: p[1]) if valid else out[0]
    print(f"L*={Ls}  val {ev*100:.2f}%  test {et*100:.2f}%")
    return out, Ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    F, sp, src = load_features(args)
    print(f"clips: {len(src)} | " + " ".join(f"{s}:{(src==s).sum()}" for s in SOURCES))

    def masks(a, b):
        m = (src == a) | (src == b)
        y = (src == a).astype(int)
        return (y, m & (sp == "train"), m & (sp == "val"), m & (sp == "test"))

    # E1 同代对决:suno vs jamendo
    y1, tr1, va1, te1 = masks("suno", "jamendo")
    c1, L1 = probe_curve(F, y1, tr1, va1, te1, "E1 同代对决:suno vs jamendo(2024-26 人类)")

    # E2 旧探针跨年代迁移:训 suno vs fma,评 suno-te vs jamendo-te
    y2, tr2, va2, te2 = masks("suno", "fma")
    c2, L2 = probe_curve(F, y2, tr2, va2, te2, "E2a 参照:suno vs fma(原域内)")
    m_x = ((src == "suno") | (src == "jamendo")) & (sp == "test")
    y_x = (src == "suno").astype(int)
    print(f"\n=== E2b 迁移:suno-vs-fma 探针 -> suno-test vs jamendo-test ===")
    print(f"{'layer':>5} {'transfer':>9}")
    es = []
    for L, ev, et, clf in c2:
        e = compute_eer(y_x[m_x], linear_clf.score(clf, F[m_x, L]))["eer"]
        es.append(e)
        print(f"{L:>5} {e*100:>8.2f}%", flush=True)
    star = es[L2]
    bL = int(np.nanargmin(es))
    print(f"@L*(={L2}) transfer EER = {star*100:.2f}% | best-layer {es[bL]*100:.2f}% (L{bL})")

    # E3 年代信号本身:fma vs jamendo(人类 vs 人类)
    y3, tr3, va3, te3 = masks("jamendo", "fma")
    c3, L3 = probe_curve(F, y3, tr3, va3, te3, "E3 年代信号:fma(2010s) vs jamendo(2024-26),人类对人类")

    e1 = c1[L1][2]; e2a = c2[L2][2]; e3 = c3[L3][2]
    print(f"\n===== 三连测判读({args.encoder})=====")
    print(f"E1 suno vs jamendo(同代):     {e1*100:6.2f}%   (近0=信号不靠年代;趋50=红旗坐实)")
    print(f"E2a suno vs fma(参照):        {e2a*100:6.2f}%   | E2b 迁移: {star*100:.2f}%(低=real定义不锚年代)")
    print(f"E3 fma vs jamendo(年代本身):  {e3*100:6.2f}%   (低=编码器里年代信号强,高=弱)")


if __name__ == "__main__":
    main()
