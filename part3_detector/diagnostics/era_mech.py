"""
人类音乐集混淆 · 机制三连(2026-08-12,Batch G 续集,era_fix 出数后立项)。

batchG 判决:混训修复成立(镜像 LOGO),副产迁移不对称(fma 训→jam 摔得重,
jam 训→fma 摔得轻)。本脚本把"为什么"钉死,三个部分:

  A 探针方向夹角:fma 探针 / jam 探针 / 语料轴(fma vs jam)三根箭头的逐层几何。
     预测:浅层 fma、jam 探针近乎重合(迁移好),深层张开;且 fma 探针在语料轴上的
     投影(偷懒掺捷径)比 jam 探针大——判决 2 的"简单考题学不出真本事"的直接证据。
  B 掺量曲线:fma 全量 + jamendo 掺 {0,100,200,400,800,全量} 首训练,
     看 worst-case 何时压平——"每个新语料需要多少疫苗"。
  C 误伤打分导出:fma-only 探针给全部 jamendo 打分(探针从未见过 jamendo,
     2050 首全可用),连同 suno/fma test 侧分数与 EER 阈值一起落 csv,
     供本地做纯统计误伤画像(按 genre/手工特征切,零 ML)。

特征缓存与 era_fix 完全共享,CPU 分钟级。

Usage(服务器,venv):
  python -u diagnostics/era_mech.py --data-dir data_store/crossgen_export --encoder muq --inst-only 2>&1 | tee batchG_mech_muq.txt
产物:stdout 表格 + era_mech_scores_<encoder>.csv(部分 C)。
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifiers import linear as linear_clf
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

SOURCES = ("suno", "fma", "jamendo")
JAM_DOSES = (0, 100, 200, 400, 800, None)  # None = 全量


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
    F, sp, src, aid = [], [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); sp.append(r["split"]); src.append(r["source"]); aid.append(r["audio_id"])
    F = np.stack(F)
    print(f"Loaded {len(aid)} clips ({args.encoder}) | "
          + " ".join(f"{s}:{sum(1 for x in src if x == s)}" for s in SOURCES), flush=True)
    return F, np.array(sp), np.array(src), np.array(aid)


def fit_dir(X, y):
    """公共标准化空间里的 LR 方向(单位向量)。调用方负责先 transform。"""
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    lr.fit(X, y)
    w = lr.coef_[0]
    return w / np.linalg.norm(w)


def part_a(F, sp, src):
    """A:三根箭头的逐层几何。语料轴 y=1 为 jamendo(指向"现代/新平台")。"""
    n_layers = F.shape[1]
    tr = sp == "train"
    print("\n=== A 探针方向夹角(公共 scaler,train 三语料并集)===")
    print(f"{'layer':>5} {'角(fma,jam)':>12} {'cos(fma,语料轴)':>16} {'cos(jam,语料轴)':>16}")
    for L in range(n_layers):
        sc = StandardScaler().fit(F[tr, L])
        X = sc.transform(F[:, L])
        m_f = tr & ((src == "suno") | (src == "fma"))
        m_j = tr & ((src == "suno") | (src == "jamendo"))
        m_e = tr & ((src == "fma") | (src == "jamendo"))
        w_f = fit_dir(X[m_f], (src[m_f] == "suno").astype(int))
        w_j = fit_dir(X[m_j], (src[m_j] == "suno").astype(int))
        w_e = fit_dir(X[m_e], (src[m_e] == "jamendo").astype(int))
        ang = np.degrees(np.arccos(np.clip(w_f @ w_j, -1, 1)))
        print(f"{L:>5} {ang:>11.1f}° {w_f @ w_e:>16.3f} {w_j @ w_e:>16.3f}", flush=True)
    print("判读:角小=两探针学的是同一根 AI 轴;fma 列 cos 显著为正而 jam 列近 0\n"
          "=> fma 探针把'现代感'掺进了判据(偷懒捷径),jam 探针被迫学纯 AI 指纹。")


def part_b(F, sp, src):
    """B:掺量曲线。fma 全量 + jamendo 掺 n 首,L* 按 mixed val 选(与 batchG 同法)。"""
    n_layers = F.shape[1]
    y_ai = (src == "suno").astype(int)
    jam_tr_idx = np.where((src == "jamendo") & (sp == "train"))[0]
    order = np.random.default_rng(0).permutation(len(jam_tr_idx))
    print(f"\n=== B 掺量曲线(fma 全量 + jamendo 掺量;jam train 池 {len(jam_tr_idx)})===")
    print(f"{'jam掺量':>8} {'L*':>4} {'vs fma':>9} {'vs jamendo':>11} {'vs both':>9} {'worst':>9}")
    m_te_f = (sp == "test") & ((src == "suno") | (src == "fma"))
    m_te_j = (sp == "test") & ((src == "suno") | (src == "jamendo"))
    m_te_b = (sp == "test") & np.isin(src, SOURCES)
    m_va = (sp == "val") & np.isin(src, SOURCES)   # 混合 val(batchG 判决 3:选层用混合 val)
    for dose in JAM_DOSES:
        take = jam_tr_idx[order[:dose]] if dose is not None else jam_tr_idx
        m_tr = (sp == "train") & ((src == "suno") | (src == "fma"))
        tr_idx = np.concatenate([np.where(m_tr)[0], take])
        best = None
        for L in range(n_layers):
            clf = linear_clf.train(F[tr_idx, L], y_ai[tr_idx], {"C": 1.0})
            ev = compute_eer(y_ai[m_va], linear_clf.score(clf, F[m_va, L]))["eer"]
            if best is None or ev < best[1]:
                best = (L, ev, clf)
        L, _, clf = best
        e_f = compute_eer(y_ai[m_te_f], linear_clf.score(clf, F[m_te_f, L]))["eer"]
        e_j = compute_eer(y_ai[m_te_j], linear_clf.score(clf, F[m_te_j, L]))["eer"]
        e_b = compute_eer(y_ai[m_te_b], linear_clf.score(clf, F[m_te_b, L]))["eer"]
        tag = str(dose) if dose is not None else f"全量{len(jam_tr_idx)}"
        print(f"{tag:>8} {L:>4} {e_f*100:>8.2f}% {e_j*100:>10.2f}% {e_b*100:>8.2f}% "
              f"{max(e_f, e_j)*100:>8.2f}%", flush=True)
    print("判读:worst 压平所需的最小掺量 = 每个新人类语料的'疫苗剂量'。")


def part_c(F, sp, src, aid, args):
    """C:fma-only 探针(从未见 jamendo)给全部 jamendo 打分,落 csv 供本地画像。"""
    n_layers = F.shape[1]
    y_ai = (src == "suno").astype(int)
    m_tr = (sp == "train") & ((src == "suno") | (src == "fma"))
    m_va = (sp == "val") & ((src == "suno") | (src == "fma"))
    curve = []
    for L in range(n_layers):
        clf = linear_clf.train(F[m_tr, L], y_ai[m_tr], {"C": 1.0})
        ev = compute_eer(y_ai[m_va], linear_clf.score(clf, F[m_va, L]))["eer"]
        curve.append((ev, L, clf))
    _, Ls, _ = min(curve)
    layers = sorted({0, Ls})
    print(f"\n=== C 误伤打分导出(fma-only 探针,层 {layers})===")
    m_te = (sp == "test") & ((src == "suno") | (src == "fma"))
    cols, thrs = {}, {}
    for L in layers:
        clf = curve[L][2]
        cols[L] = linear_clf.score(clf, F[:, L])
        r = compute_eer(y_ai[m_te], cols[L][m_te])
        thrs[L] = r["threshold"]
        m_jam = src == "jamendo"
        hurt = float((cols[L][m_jam] > r["threshold"]).mean())
        print(f"  L{L}: suno-vs-fma test EER {r['eer']*100:.2f}%(阈值 {r['threshold']:.4f})"
              f" | jamendo 误伤率 {hurt*100:.1f}%", flush=True)
    out = Path(f"era_mech_scores_{args.encoder}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["audio_id", "source", "split"]
                   + [f"score_L{L}" for L in layers] + [f"flag_ai_L{L}" for L in layers])
        for i in range(len(aid)):
            w.writerow([aid[i], src[i], sp[i]]
                       + [f"{cols[L][i]:.6f}" for L in layers]
                       + [int(cols[L][i] > thrs[L]) for L in layers])
    print(f"  全部 {len(aid)} 行(含三语料)写入 {out};阈值取 suno-vs-fma test EER 点。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--parts", default="ABC", help="要跑的部分,如 AB")
    ap.add_argument("--inst-only", action="store_true",
                    help="jamendo 只用纯器乐 2050 首(口径规定:SSL 涉 jamendo 一律开)")
    args = ap.parse_args()
    F, sp, src, aid = load(args)
    if "A" in args.parts:
        part_a(F, sp, src)
    if "B" in args.parts:
        part_b(F, sp, src)
    if "C" in args.parts:
        part_c(F, sp, src, aid, args)


if __name__ == "__main__":
    main()
