"""
人类音乐集混淆 · genre 两连(2026-08-13,Batch H;弹药库 §6 第 5/6 发)。

背景:Batch G-mech 误伤画像显示 genre 梯度(blues 24.6% vs classical 5.0%),
稀缺度核验"方向对但不单调"(pop/rock 稀缺却不被冤)——真正变量疑似"声学覆盖"。
本脚本把 genre 泛化轴从观察变测量:

  L genre-LOGO:suno vs jamendo(同代对决,--inst-only),8 genre 轮流留一考一。
     训练与选层都不见留出 genre;评测用该 genre 的全部行(探针从未见过,无泄漏)。
     held-out EER 显著高于 seen EER => genre 泛化轴坐实(轴③从观察变测量);
     幅度按 genre 排序 = 声学覆盖地图。
  V 疫苗配方:fma 全量 + 100 首 jamendo,两种配方 × 3 个种子:
     electronic100(全掺 electronic)vs spread100(8 genre 各 ~12 首)。
     spread 显著更低 => "疫苗配方要按 genre 摊开"成立。

genre 来源:jamendo 写在 audio_id(jam_<genre>_NNNN);suno 用随脚本走的
suno_genres.csv(来自 round1 manifest,8 类均衡 318–440)。
特征缓存与 era_fix 共享,CPU 约 10–20 分钟/编码器。

Usage(服务器,venv):
  python -u diagnostics/era_genre.py --data-dir data_store/crossgen_export --encoder muq --inst-only 2>&1 | tee batchH_genre_muq.txt
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import linear as linear_clf
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

GENRES = ("blues", "classical", "country", "electronic", "hiphop", "jazz", "pop", "rock")
SOURCES = ("suno", "fma", "jamendo")
VACCINE_N = 100
SEEDS = (0, 1, 2)


def load(args):
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
    sg_path = Path(os.path.dirname(os.path.abspath(__file__))) / "suno_genres.csv"
    suno_genre = {r["audio_id"]: r["genre"] for r in csv.DictReader(open(sg_path))}
    F, sp, src, gen = [], [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if not p.exists():
            continue
        if r["source"] == "jamendo":
            g = r["audio_id"].split("_")[1]
        elif r["source"] == "suno":
            g = suno_genre.get(r["audio_id"], "")
        else:
            g = ""      # fma 不参与 genre 分层
        F.append(np.load(p)); sp.append(r["split"]); src.append(r["source"]); gen.append(g)
    F = np.stack(F); sp = np.array(sp); src = np.array(src); gen = np.array(gen)
    print(f"Loaded {len(src)} clips ({args.encoder}) | "
          + " ".join(f"{s}:{(src==s).sum()}" for s in SOURCES), flush=True)
    print("suno genre 覆盖: " + " ".join(
        f"{g}:{((src=='suno')&(gen==g)).sum()}" for g in GENRES), flush=True)
    return F, sp, src, gen


def eval_at(clf, F, y, m, L):
    return compute_eer(y[m], linear_clf.score(clf, F[m, L]))["eer"]


def part_logo(F, sp, src, gen):
    """L:suno vs jamendo 留一 genre。评测用留出 genre 的全部行(从未参与训练/选层)。"""
    n_layers = F.shape[1]
    m_sj = (src == "suno") | (src == "jamendo")
    y = (src == "suno").astype(int)
    print("\n=== L genre-LOGO(suno vs jamendo,训 7 考 1)===")
    print(f"{'留出':>12} {'L*':>4} {'n_held(s/j)':>12} {'seen te':>9} {'held-out':>9} {'Δ':>7}")
    seen_all, held_all = [], []
    for g in GENRES:
        m_in = m_sj & (gen != g)
        tr = m_in & (sp == "train")
        va = m_in & (sp == "val")
        te = m_in & (sp == "test")
        m_held = m_sj & (gen == g)
        best = None
        for L in range(n_layers):
            clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
            ev = eval_at(clf, F, y, va, L)
            if best is None or ev < best[1]:
                best = (L, ev, clf)
        L, _, clf = best
        e_seen = eval_at(clf, F, y, te, L)
        e_held = eval_at(clf, F, y, m_held, L)
        seen_all.append(e_seen); held_all.append(e_held)
        ns = int((m_held & (src == "suno")).sum()); nj = int((m_held & (src == "jamendo")).sum())
        print(f"{g:>12} {L:>4} {ns:>6}/{nj:<5} {e_seen*100:>8.2f}% {e_held*100:>8.2f}% "
              f"{(e_held-e_seen)*100:>+6.2f}", flush=True)
    print(f"{'均值':>12} {'':>4} {'':>12} {np.mean(seen_all)*100:>8.2f}% "
          f"{np.mean(held_all)*100:>8.2f}% {(np.mean(held_all)-np.mean(seen_all))*100:>+6.2f}")
    print("判读:held-out 显著高于 seen => genre 泛化轴坐实;逐 genre 的 Δ = 声学覆盖地图"
          "(预测:blues/country/jazz 大,electronic/classical 小)。")


def part_vaccine(F, sp, src, gen):
    """V:fma 全量 + 100 首 jamendo,electronic100 vs spread100,各 3 种子。"""
    n_layers = F.shape[1]
    y = (src == "suno").astype(int)
    m_va = (sp == "val") & np.isin(src, SOURCES)          # 混合 val 选层(batchG 判决 3)
    m_te_f = (sp == "test") & ((src == "suno") | (src == "fma"))
    m_te_j = (sp == "test") & ((src == "suno") | (src == "jamendo"))
    base_tr = np.where((sp == "train") & ((src == "suno") | (src == "fma")))[0]
    jam_tr = np.where((src == "jamendo") & (sp == "train"))[0]

    def run(tr_idx, tag):
        best = None
        for L in range(n_layers):
            clf = linear_clf.train(F[tr_idx, L], y[tr_idx], {"C": 1.0})
            ev = eval_at(clf, F, y, m_va, L)
            if best is None or ev < best[1]:
                best = (L, ev, clf)
        L, _, clf = best
        e_f = eval_at(clf, F, y, m_te_f, L)
        e_j = eval_at(clf, F, y, m_te_j, L)
        print(f"{tag:>16} {L:>4} {e_f*100:>8.2f}% {e_j*100:>10.2f}% {max(e_f,e_j)*100:>8.2f}%",
              flush=True)
        return max(e_f, e_j)

    print(f"\n=== V 疫苗配方(fma 全量 + {VACCINE_N} 首 jamendo × 3 种子)===")
    print(f"{'配方':>16} {'L*':>4} {'vs fma':>9} {'vs jamendo':>11} {'worst':>9}")
    run(base_tr, "dose-0 参照")
    results = {"electronic100": [], "spread100": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pool_e = jam_tr[gen[jam_tr] == "electronic"]
        take_e = rng.choice(pool_e, size=min(VACCINE_N, len(pool_e)), replace=False)
        results["electronic100"].append(
            run(np.concatenate([base_tr, take_e]), f"electronic100 s{seed}"))
        picks = []
        per = int(np.ceil(VACCINE_N / len(GENRES)))
        for g in GENRES:
            pool_g = jam_tr[gen[jam_tr] == g]
            picks.append(rng.choice(pool_g, size=min(per, len(pool_g)), replace=False))
        picks = np.concatenate(picks)
        rng.shuffle(picks)          # 审计修复:不 shuffle 则按 GENRES 顺序截断,rock 恒少配额
        picks = picks[:VACCINE_N]
        results["spread100"].append(
            run(np.concatenate([base_tr, picks]), f"spread100 s{seed}"))
    for tag, es in results.items():
        print(f"{tag:>16} worst 均值 {np.mean(es)*100:.2f}%(范围 "
              f"{min(es)*100:.2f}–{max(es)*100:.2f}%)")
    print("判读:spread 显著低于 electronic => 疫苗配方要按 genre 摊开;"
          "两者接近 => 100 首的作用主要是'来源味',genre 覆盖靠边。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--parts", default="LV", help="要跑的部分,如 L")
    ap.add_argument("--inst-only", action="store_true",
                    help="jamendo 只用纯器乐 2050 首(口径规定:SSL 涉 jamendo 一律开)")
    args = ap.parse_args()
    F, sp, src, gen = load(args)
    if "L" in args.parts:
        part_logo(F, sp, src, gen)
    if "V" in args.parts:
        part_vaccine(F, sp, src, gen)


if __name__ == "__main__":
    main()
