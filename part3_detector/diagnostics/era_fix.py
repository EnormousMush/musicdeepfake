"""
real 侧语料混淆 · 混训修复实验(2026-08-10,batchE 续集)。

batchE E2b 发现:fma 单独当"人类"训的探针,换到现代人类(jamendo)上退化到 7-14%
——生成器混淆的镜像(泛化第二根轴:real 侧)。假侧的解法是多生成器混训
(LOGO 44.9%→8.3%);本实验验证镜像解法:real 侧 fma+jamendo 混训。

三种训练配方 × 三个测试对,逐层线性探针(同老配方):
  训 fma-only   / jam-only / mixed(fma+jam)
  测 suno-te vs fma-te | suno-te vs jamendo-te | suno-te vs 两者并集
判读:mixed 的最差单测(worst-case)若压回 ~2% 以内,镜像修复成立;
jam-only 行顺带回答反方向迁移(现代人类训的探针认不认 2010s 人类)。

Usage(服务器,特征已缓存,CPU 分钟级):
  python -u diagnostics/era_fix.py --data-dir data_store/crossgen_export --encoder muq 2>&1 | tee batchF_muq.txt
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

SOURCES = ("suno", "fma", "jamendo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--inst-only", action="store_true",
                    help="jamendo 只用纯器乐 2050 首(与手工特征侧口径一致)")
    args = ap.parse_args()

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
    F = np.stack(F); sp = np.array(sp); src = np.array(src)
    n_layers = F.shape[1]
    print(f"Loaded {len(src)} clips ({args.encoder}) | "
          + " ".join(f"{s}:{(src==s).sum()}" for s in SOURCES), flush=True)

    y_ai = (src == "suno").astype(int)
    RECIPES = {"fma-only": ["fma"], "jam-only": ["jamendo"], "mixed": ["fma", "jamendo"]}
    TESTS = {"vs fma": ["fma"], "vs jamendo": ["jamendo"], "vs both": ["fma", "jamendo"]}

    def mask(sources, split):
        m = (sp == split) & ((src == "suno") | np.isin(src, sources))
        return m

    summary = {}
    for rname, real_srcs in RECIPES.items():
        tr = mask(real_srcs, "train")
        va = mask(real_srcs, "val")
        print(f"\n=== 训练配方 [{rname}](train {tr.sum()})===", flush=True)
        print(f"{'layer':>5} {'val':>8} | " + " ".join(f"{t:>12}" for t in TESTS))
        per_layer = []
        for L in range(n_layers):
            clf = linear_clf.train(F[tr, L], y_ai[tr], {"C": 1.0})
            ev = compute_eer(y_ai[va], linear_clf.score(clf, F[va, L]))["eer"]
            tests = {}
            for tname, tsrcs in TESTS.items():
                m = mask(tsrcs, "test")
                tests[tname] = compute_eer(y_ai[m], linear_clf.score(clf, F[m, L]))["eer"]
            per_layer.append((ev, tests))
            print(f"{L:>5} {ev*100:>7.2f}% | "
                  + " ".join(f"{tests[t]*100:>11.2f}%" for t in TESTS), flush=True)
        Ls = int(np.nanargmin([p[0] for p in per_layer]))
        summary[rname] = (Ls, per_layer[Ls][1])
        print(f"L*={Ls}(val {per_layer[Ls][0]*100:.2f}%)")

    print(f"\n===== 汇总 @各自 L*({args.encoder})=====")
    print(f"{'训练配方':>10} {'L*':>4} | " + " ".join(f"{t:>12}" for t in TESTS) + f" {'worst':>9}")
    for rname, (Ls, tests) in summary.items():
        worst = max(tests[t] for t in ("vs fma", "vs jamendo"))
        print(f"{rname:>10} {Ls:>4} | "
              + " ".join(f"{tests[t]*100:>11.2f}%" for t in TESTS)
              + f" {worst*100:>8.2f}%")
    print("\n判读:mixed 行 worst-case 压回 ~2% 内 => real 侧混训修复成立(镜像 LOGO);"
          "\njam-only 行的 'vs fma' = 反方向迁移(现代训的探针认不认旧人类)。", flush=True)


if __name__ == "__main__":
    main()
