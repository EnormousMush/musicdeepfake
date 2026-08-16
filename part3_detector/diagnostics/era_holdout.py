"""
第三来源 held-out 终审(2026-08-16,Batch I;人类音乐集混淆 §6 第 7 发收官)。

问题:混训修复(fma+jamendo)在见过的两个语料上及格,只证明"认得见过的两家"。
本实验拿两个训练从未见过的人类来源当考卷——ccMixter 365(2020–26)+
IA netlabels 450(2024–26),共同规格 10s,**只测不训**:

  训练配方(同 era_fix 三件套):fma-only / jam-only / mixed(fma+jam)
  考卷:suno-te vs {fma-te, jam-te}(参照)+ suno-te vs {ccmixter, ianet, 拼盘}(终审)
  选层:混合 val(batchG 判决 3)
判读:mixed 在拼盘上 ≈ 它在 jam 上的水平(~2–5%)=> 混训学到的是"认人类",
修复结论可进论文;大幅劣化(>10%)=> 背题,milestone 重开。
副产:heldout 全体打分落 csv(带 source/year/artist 可回联),本地做
分来源/分年份/按创作者聚类的稳健性附检(纯统计)。

Usage(服务器;heldout 特征首跑现算,GPU ~5min / CPU 慢跑亦可):
  python -u diagnostics/era_holdout.py --data-dir data_store/crossgen_export \
    --heldout-dir data_store/heldout_export --encoder muq --inst-only 2>&1 | tee batchI_holdout_muq.txt
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import linear as linear_clf
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

SOURCES = ("suno", "fma", "jamendo")


def load_crossgen(args):
    data_dir = Path(args.data_dir)
    cache = data_dir / "features" / args.encoder
    rows = [r for r in csv.DictReader(open(data_dir / "manifest.csv"))
            if r["source"] in SOURCES]
    if args.inst_only:
        ids = instrumental_ids()
        if ids is None:
            print("警告:未找到 jamendo_instrumental.txt,未过滤人声", flush=True)
        else:
            n0 = sum(1 for r in rows if r["source"] == "jamendo")
            rows = filter_rows(rows, ids)
            print(f"inst-only: jamendo {n0} -> {sum(1 for r in rows if r['source']=='jamendo')}", flush=True)
    F, sp, src = [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); sp.append(r["split"]); src.append(r["source"])
    print(f"crossgen: {len(src)} clips | "
          + " ".join(f"{s}:{sum(1 for x in src if x==s)}" for s in SOURCES), flush=True)
    return np.stack(F), np.array(sp), np.array(src)


def load_heldout(args):
    hdir = Path(args.heldout_dir)
    cache = hdir / "features" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(hdir / "manifest.csv")))
    enc, F, meta, failures = None, [], [], []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cpath = cache / f"{r['audio_id']}.npy"
        try:
            if cpath.exists():
                f = np.load(cpath)
            else:
                if enc is None:
                    from encoders.ssl import SSLEncoder
                    enc = SSLEncoder(args.encoder, device=args.device)
                wav, srate = sf.read(hdir / r["rel_path"])
                f = enc.encode_all_layers(np.asarray(wav, dtype=np.float32), srate)
                np.save(cpath, f)
            F.append(f); meta.append(r)
        except Exception as e:
            failures.append((r["audio_id"], repr(e)))
        if i % 200 == 0 or i == len(rows):
            print(f"  heldout {i}/{len(rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)", flush=True)
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")
    hsrc = np.array([m["source"] for m in meta])
    print(f"heldout: {len(meta)} clips | "
          + " ".join(f"{s}:{(hsrc==s).sum()}" for s in sorted(set(hsrc))), flush=True)
    return np.stack(F), hsrc, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--heldout-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--device", default=None)
    ap.add_argument("--inst-only", action="store_true",
                    help="jamendo 只用纯器乐 2050(口径规定)")
    args = ap.parse_args()

    F, sp, src = load_crossgen(args)
    H, hsrc, hmeta = load_heldout(args)
    n_layers = F.shape[1]
    y = (src == "suno").astype(int)
    m_va = (sp == "val")                                   # 混合 val 选层
    m_su_te = (src == "suno") & (sp == "test")
    m_te = {"vs fma": (sp == "test") & ((src == "suno") | (src == "fma")),
            "vs jamendo": (sp == "test") & ((src == "suno") | (src == "jamendo"))}
    HO = {"vs ccmixter": hsrc == "ccmixter", "vs ianet": hsrc == "ianet",
          "vs 拼盘": np.ones(len(hsrc), bool)}
    RECIPES = {"fma-only": ["fma"], "jam-only": ["jamendo"], "mixed": ["fma", "jamendo"]}

    dump = {}
    print(f"\n{'配方':>10} {'L*':>4} {'vs fma':>9} {'vs jamendo':>11} "
          f"{'vs ccmixter':>12} {'vs ianet':>10} {'vs 拼盘':>9}")
    for rname, reals in RECIPES.items():
        tr = (sp == "train") & ((src == "suno") | np.isin(src, reals))
        best = None
        for L in range(n_layers):
            clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
            ev = compute_eer(y[m_va], linear_clf.score(clf, F[m_va, L]))["eer"]
            if best is None or ev < best[1]:
                best = (L, ev, clf)
        L, _, clf = best
        s_su = linear_clf.score(clf, F[m_su_te, L])
        s_ho = linear_clf.score(clf, H[:, L])
        es = []
        for tag, m in m_te.items():
            es.append(compute_eer(y[m], linear_clf.score(clf, F[m, L]))["eer"])
        for tag, m in HO.items():
            yy = np.concatenate([np.ones(int(m_su_te.sum())), np.zeros(int(m.sum()))])
            ss = np.concatenate([s_su, s_ho[m]])
            es.append(compute_eer(yy, ss)["eer"])
        print(f"{rname:>10} {L:>4} " + " ".join(f"{e*100:>{w}.2f}%" for e, w in
              zip(es, (8, 10, 11, 9, 8))), flush=True)
        dump[rname] = (L, s_ho)

    out = Path(f"era_holdout_scores_{args.encoder}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["audio_id", "source", "year", "score_fma_only", "score_jam_only", "score_mixed"])
        for i, m in enumerate(hmeta):
            w.writerow([m["audio_id"], m["source"], m.get("year", "")]
                       + [f"{dump[r][1][i]:.6f}" for r in RECIPES])
    print(f"\nheldout 打分已写入 {out}(mixed L*={dump['mixed'][0]});"
          "\n判读:mixed 行的拼盘成绩 ≈ 它的 vs jamendo 水平 => 修复真泛化;"
          "\n大幅劣化 => 背题。fma-only 行 = 未修复对照(预期在拼盘上也退化)。", flush=True)


if __name__ == "__main__":
    main()
