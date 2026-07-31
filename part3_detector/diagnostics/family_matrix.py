"""
家族×家族转移矩阵 — 三分支合一的主战实验(2026-07-31 立项)。

一张"训练族 × 测试生成器"的 EER 表,同时回答:
  ① 同族结论鲁棒性:反方向训练也聚成块 -> 家族结构对称、真实;
  ② 退化测试:最后一行"全池"vs 各族专职行,逐生成器读退化量;
  ③ 编码器共振:本脚本按 --encoder 各跑一遍(mert/muq/wav2vec2),交互模式对比。

正式协议(相对旧流程的两个硬化):
  - 训练量配平:每个训练池的假货侧下采样到同一 n(--n-train),杀掉池间数据量混淆;
  - bootstrap 95% CI:每个格子重抽 --boot 次,报区间,比较不再靠裸眼。
评测统一用各生成器的 20% 留出份(suno 用原 test 份),训练池内外同卷,格子间可比。

Usage(服务器,特征缓存已存在):
  python diagnostics/family_matrix.py --data-dir data_store/crossgen_export --encoder muq
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer

# 架构族定义(随版图更新;mureka 为商用盲盒,单列成行看它像谁)
FAMILIES = {
    "suno_fam(LM+连续渲染)": ["suno", "acestep", "levo"],
    "dr_fam(流匹配直渲)": ["dr1", "diffrhythm2"],
    "lofi_diff(老潜扩散)": ["audioldm2", "musicldm", "mustango"],
    "ar_lm(纯AR)": ["MusicGen_medium"],
    "dit(潜DiT)": ["stable_audio_open"],
    "token_lm(离散token+FM超分)": ["inspire"],
    "mureka(商用盲盒)": ["mureka"],
}


def bootstrap_eer(y, s, n_boot, rng):
    point = compute_eer(y, s)["eer"]
    if n_boot <= 0:
        return point, point, point
    idx_r = np.where(y == 0)[0]
    idx_f = np.where(y == 1)[0]
    boots = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(idx_r, len(idx_r), replace=True),
                             rng.choice(idx_f, len(idx_f), replace=True)])
        boots.append(compute_eer(y[bi], s[bi])["eer"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=800,
                    help="每个训练池假货侧配平到的数量(全池行也配平到此数)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap 次数;0 关闭")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    cache = data_dir / "features" / args.encoder

    F, y, sp, src = [], [], [], []
    for r in rows:
        p = cache / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); y.append(int(r["label"]))
            sp.append(r["split"]); src.append(r["source"])
    F = np.stack(F); y = np.array(y); sp = np.array(sp); src = np.array(src)
    n_layers = F.shape[1]
    gens = sorted(set(src) - {"fma"})
    print(f"Loaded {len(y)} clips, {n_layers} layers ({args.encoder})")
    present = {g for fam in FAMILIES.values() for g in fam}
    missing = [g for g in gens if g not in present]
    if missing:
        print(f"⚠️ 未入族的生成器(不参与训练,仍被评测): {missing}")

    # 角色划分:每个生成器 80/20(suno 用原 split;fma 用原 split)
    rng = np.random.default_rng(args.seed)
    role = np.array(["-"] * len(y), dtype=object)
    role[(src == "fma") & (sp == "train")] = "train"
    role[(src == "fma") & (sp == "test")] = "eval"
    for g in gens:
        idx = np.where(src == g)[0]
        if g == "suno":
            role[idx[sp[idx] == "train"]] = "train"
            role[idx[sp[idx] == "test"]] = "eval"
        else:
            perm = rng.permutation(idx)
            cut = int(0.8 * len(perm))
            role[perm[:cut]] = "train"
            role[perm[cut:]] = "eval"

    fma_train_idx = np.where((src == "fma") & (role == "train"))[0]
    fma_eval_idx = np.where((src == "fma") & (role == "eval"))[0]
    gen_eval_idx = {g: np.where((src == g) & (role == "eval"))[0] for g in gens}

    pools = dict(FAMILIES)
    pools["全池(退化行)"] = [g for fam in FAMILIES.values() for g in fam if g in gens]

    boot_rng = np.random.default_rng(args.seed + 1)
    for pool_name, members in pools.items():
        members = [g for g in members if g in gens]
        if not members:
            print(f"\n### {pool_name}: 成员缺席,跳过")
            continue
        fake_train_idx = np.concatenate(
            [np.where((src == g) & (role == "train"))[0] for g in members])
        # 训练量配平(池间公平的命门)
        if len(fake_train_idx) > args.n_train:
            fake_train_idx = rng.choice(fake_train_idx, args.n_train, replace=False)
        elif len(fake_train_idx) < args.n_train:
            print(f"⚠️ [{pool_name}] 只有 {len(fake_train_idx)} < 配平目标 {args.n_train},配平失效")
        tr = np.concatenate([fma_train_idx, fake_train_idx])
        # 选层:池内 seen-eval(成员 eval + fma-eval),与 LOGO 协议一致
        seen_eval = np.concatenate([gen_eval_idx[g] for g in members] + [fma_eval_idx])
        best = None
        for L in range(n_layers):
            clf = linear_clf.train(F[tr, L], y[tr], {"C": 1.0})
            e = compute_eer(y[seen_eval], linear_clf.score(clf, F[seen_eval, L]))["eer"]
            if best is None or e < best[1]:
                best = (L, e, clf)
        L, e_seen, clf = best
        print(f"\n### 训练池 [{pool_name}] 成员={members} 配平后假货n={len(fake_train_idx)} "
              f"L*={L} seen-eval={e_seen*100:.2f}%")
        print(f"{'测试生成器':>20} {'EER':>8} {'95% CI':>18} {'池内?':>5}")
        for g in gens:
            ev = np.concatenate([fma_eval_idx, gen_eval_idx[g]])
            pt, lo, hi = bootstrap_eer(y[ev], linear_clf.score(clf, F[ev, L]),
                                       args.boot, boot_rng)
            mark = "内" if g in members else ""
            print(f"{g:>20} {pt*100:>7.2f}% [{lo*100:>6.2f}, {hi*100:>6.2f}]% {mark:>5}",
                  flush=True)

    print("\n判读:① 块对角(族内低/跨族高)且反方向成立 -> 家族结构对称、真实;"
          "\n② 全池行 vs 各族行逐列对比 = 退化量(CI 分开才算真退化);"
          "\n③ mureka 行/列的抓取模式 = 商用盲盒的架构考古读数;"
          "\n④ 本表按编码器各跑一遍,交互模式差异喂给编码器共振分支。")


if __name__ == "__main__":
    main()
