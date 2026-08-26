"""入档演习(议题二决胜实验,2026-08-26 预注册;实验2论文方向 议题二节)。

剧本:馆里先只有人类参考,12 家生成器按时间顺序逐个入档;每步量五指标
(入档成本/严误报战力/遗忘/样本效率/未知桶聚类)。本脚本只负责服务器侧的
**分数倾倒**(针层 + 梯度对照组),阈值校准/Šidák/融合/聚类全在 Mac 本地做
(倾倒式 = 分析可反复迭代,不用重跑服务器)。

针层(archive):每家 × 每档样本预算(100/300/1000)训一根针
(该家 pool vs fma probe-train 的凸 LR,与 angle_matrix 同法),
对全体 cal+eval 片段倾倒 logit;记录入档墙钟。**入档=追加,互不影响**。

对照组(budget=300,时间顺序连续剧,单一"AI vs 人类"二分头):
  retrain  = 每步全量重训(已入档全家 pool + fma);
  finetune = 上一步权重继续,只喂新家+fma(裸微调,预期灾难遗忘);
  replay   = finetune + 每家留 50 首回放缓冲;
  ewc      = finetune + Fisher 对角惩罚(lam=100);
  noupdate = 第一步训完永不更新。
每步对 cal+eval 全体倾倒分数。

划分(md5 确定性):各家 1000 首 test → <80 pool / >=80 eval;suno 用
train=pool、test=eval。fma train 2100 → <67 probe-train / else cal;
jamendo train+val 全部 cal;eval 人类 = fma test + jamendo test + ccmixter + ianet。
层选择:历史口径,单层头按 suno/fma val EER 选 L*,全场共用。

Usage(服务器 .venv,tmux,纯 CPU 亦可):
  python diagnostics/enrollment_drill.py --data-dir data_store/crossgen_export \
    --heldout-dir data_store/heldout_export --encoder muq --out-dir "$WORK/results"
"""
import argparse
import csv
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from classifiers import linear as linear_clf
from eval.eer import compute_eer
from jam_inst import instrumental_ids, filter_rows

# 入档顺序 = 发布时间顺序(2023 -> 2025+)
GENS = ["MusicGen_medium", "audioldm2", "musicldm", "mustango",
        "stable_audio_open", "inspire", "dr1", "diffrhythm2",
        "acestep", "levo", "mureka", "suno"]
BUDGETS = [100, 300, 1000]
BASE_BUDGET = 300
SEED = 20260826


def h100(aid):
    return int(hashlib.md5(aid.encode()).hexdigest()[:8], 16) % 100


def load_bed(args):
    rows = []
    for d in [args.data_dir, args.heldout_dir]:
        dd = Path(d)
        for r in csv.DictReader(open(dd / "manifest.csv")):
            r["_cache"] = dd / "features" / args.encoder
            rows.append(r)
    ids = instrumental_ids()
    if ids is not None:
        rows = filter_rows(rows, ids)
    F, meta = [], []
    for r in rows:
        p = r["_cache"] / f"{r['audio_id']}.npy"
        if p.exists():
            F.append(np.load(p)); meta.append(r)
    F = np.stack(F).astype(np.float32)
    src = np.array([m["source"] for m in meta])
    sp = np.array([m["split"] for m in meta])
    aid = np.array([m["audio_id"] for m in meta])
    print(f"bed: {F.shape} | " + " ".join(f"{s}:{(src==s).sum()}" for s in sorted(set(src))), flush=True)
    return F, src, sp, aid


def assign_roles(src, sp, aid, pile=False):
    """role: probe_train / cal / eval_human / pool / eval_ai / -
    pile=True:针的阴性侧用人类拼盘(fma+jamendo),jamendo train 对半分给
    probe_train/cal(年代教训:阴性对照只有 fma 会把 real 侧混淆刻进针里)。"""
    role = np.array(["-"] * len(src), dtype=object)
    hh = np.array([h100(a) for a in aid])
    m = (src == "fma") & (sp == "train")
    role[m & (hh < 67)] = "probe_train"
    role[m & (hh >= 67)] = "cal"
    mj = (src == "jamendo") & np.isin(sp, ["train", "val"])
    if pile:
        role[mj & (hh < 50)] = "probe_train"
        role[mj & (hh >= 50)] = "cal"
    else:
        role[mj] = "cal"
    role[(src == "fma") & (sp == "test")] = "eval_human"
    role[(src == "jamendo") & (sp == "test")] = "eval_human"
    role[np.isin(src, ["ccmixter", "ianet"])] = "eval_human"
    for g in GENS:
        m = src == g
        if g == "suno":
            role[m & (sp == "train")] = "pool"
            role[m & (sp == "test")] = "eval_ai"
        else:
            role[m & (hh < 80)] = "pool"
            role[m & (hh >= 80)] = "eval_ai"
    for r in ["probe_train", "cal", "eval_human", "pool", "eval_ai"]:
        print(f"  {r}: {(role==r).sum()}", flush=True)
    return role


def pick_layer(F, src, sp, role):
    """历史口径:suno vs fma 单层扫描,val EER 选 L*。"""
    tr = (role == "probe_train") | ((src == "suno") & (sp == "train"))
    va = ((src == "fma") | (src == "suno")) & (sp == "val")
    ytr = (src[tr] != "fma").astype(int)
    yva = (src[va] != "fma").astype(int)
    best = None
    for L in range(F.shape[1]):
        clf = linear_clf.train(F[tr, L], ytr, {"C": 1.0})
        e = compute_eer(yva, clf.decision_function(F[va, L]))["eer"]
        if best is None or e < best[1]:
            best = (L, e)
    print(f"L* = {best[0]} (suno val EER {best[1]*100:.2f}%)", flush=True)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--heldout-dir", required=True)
    ap.add_argument("--encoder", default="muq")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pile", action="store_true", help="针阴性侧用人类拼盘(fma+jam)")
    ap.add_argument("--needle-only", action="store_true", help="只跑针层(跳过对照组)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    sfx = "_pile" if args.pile else ""

    F, src, sp, aid = load_bed(args)
    role = assign_roles(src, sp, aid, pile=args.pile)
    L = pick_layer(F, src, sp, role)
    X = F[:, L]

    dump = np.isin(role, ["cal", "eval_human", "eval_ai"])
    dump_idx = np.where(dump)[0]
    ptr = np.where(role == "probe_train")[0]

    out_dir = Path(os.path.expandvars(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 针层:每家 × 每档预算,倾倒 logit ----------
    cols = {}
    timing = []
    for g in GENS:
        pool = np.where((src == g) & (role == "pool"))[0]
        for B in BUDGETS:
            take = pool if len(pool) <= B else rng.choice(pool, B, replace=False)
            tr = np.concatenate([ptr, take])
            y = (src[tr] == g).astype(int)
            t0 = time.time()
            clf = linear_clf.train(X[tr], y, {"C": 1.0})
            dt = time.time() - t0
            cols[f"{g}__b{B}"] = clf.decision_function(X[dump_idx])
            timing.append((g, B, len(take), dt))
            print(f"[needle] {g} b={B} ({len(take)} clips) {dt:.1f}s", flush=True)

    with open(out_dir / f"drill_needle_logits{sfx}.csv", "w", newline="") as f:
        w = csv.writer(f)
        names = list(cols.keys())
        w.writerow(["audio_id", "source", "role"] + names)
        for k, i in enumerate(dump_idx):
            w.writerow([aid[i], src[i], role[i]] + [f"{cols[n][k]:.5f}" for n in names])
    with open(out_dir / f"drill_timing{sfx}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gen", "budget", "n", "seconds"])
        w.writerows(timing)
    if args.needle_only:
        print("DRILL-DONE", flush=True)
        return

    # ---------- 对照组:budget=300 时间顺序连续剧(线性头,CPU 足够;
    # 共享 GPU 上 cublas 会炸,强制 CPU) ----------
    device = "cpu"
    mu, sd = X[ptr].mean(0, keepdims=True), X[ptr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    Xt = torch.from_numpy(Xn).to(device)

    pools300 = {}
    for g in GENS:
        pool = np.where((src == g) & (role == "pool"))[0]
        pools300[g] = pool if len(pool) <= BASE_BUDGET else rng.choice(pool, BASE_BUDGET, replace=False)

    def fit(net, idx_pos, idx_neg, epochs=60, ewc=None, lr=1e-2):
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        idx = np.concatenate([idx_pos, idx_neg])
        yy = torch.from_numpy((src[idx] != "fma").astype(np.float32)).to(device)
        xx = Xt[idx]
        pw = torch.tensor([max(1.0, len(idx_neg) / max(1, len(idx_pos)))], device=device)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        for _ in range(epochs):
            opt.zero_grad()
            loss = lossf(net(xx).squeeze(-1), yy)
            if ewc is not None:
                fisher, star = ewc
                for (n_, p), f_, s_ in zip(net.named_parameters(), fisher, star):
                    loss = loss + 100.0 * (f_ * (p - s_) ** 2).sum()
            loss.backward(); opt.step()
        return net

    def fisher_diag(net, idx):
        yy = torch.from_numpy((src[idx] != "fma").astype(np.float32)).to(device)
        lossf = nn.BCEWithLogitsLoss()
        net.zero_grad()
        loss = lossf(net(Xt[idx]).squeeze(-1), yy)
        loss.backward()
        return [p.grad.detach() ** 2 for p in net.parameters()]

    def newnet():
        torch.manual_seed(SEED)
        return nn.Linear(X.shape[1], 1).to(device)

    methods = ["retrain", "finetune", "replay", "ewc", "noupdate"]
    nets = {m: newnet() for m in methods}
    ewc_state, replay_buf = None, []
    bcols, bt = {}, []
    for t, g in enumerate(GENS, 1):
        new = pools300[g]
        for m in methods:
            t0 = time.time()
            if m == "retrain":
                allpos = np.concatenate([pools300[x] for x in GENS[:t]])
                nets[m] = fit(newnet(), allpos, ptr)
            elif m == "noupdate":
                if t == 1:
                    nets[m] = fit(newnet(), new, ptr)
            elif m == "finetune":
                nets[m] = fit(nets[m], new, ptr)
            elif m == "replay":
                pos = np.concatenate([new] + [b for b in replay_buf]) if replay_buf else new
                nets[m] = fit(nets[m], pos, ptr)
            elif m == "ewc":
                nets[m] = fit(nets[m], new, ptr, ewc=ewc_state)
            bt.append((m, t, g, time.time() - t0))
            with torch.no_grad():
                bcols[f"{m}__t{t:02d}"] = nets[m](Xt[dump_idx]).squeeze(-1).cpu().numpy()
        replay_buf.append(rng.choice(new, min(50, len(new)), replace=False))
        f_new = fisher_diag(nets["ewc"], np.concatenate([new, ptr]))
        star = [p.detach().clone() for p in nets["ewc"].parameters()]
        if ewc_state is None:
            ewc_state = (f_new, star)
        else:
            ewc_state = ([a + b for a, b in zip(ewc_state[0], f_new)], star)
        print(f"[baseline] step {t} ({g}) done", flush=True)

    with open(out_dir / "drill_baseline_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        names = list(bcols.keys())
        w.writerow(["audio_id", "source", "role"] + names)
        for k, i in enumerate(dump_idx):
            w.writerow([aid[i], src[i], role[i]] + [f"{bcols[n][k]:.5f}" for n in names])
    with open(out_dir / "drill_baseline_timing.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "step", "gen", "seconds"])
        w.writerows(bt)

    # ---------- 未知桶原料:eval_ai 全体在 L* 的标准化特征,本地聚类 ----------
    np.save(out_dir / "drill_evalai_feats.npy", Xn[np.where(role == "eval_ai")[0]])
    with open(out_dir / "drill_evalai_meta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["audio_id", "source"])
        for i in np.where(role == "eval_ai")[0]:
            w.writerow([aid[i], src[i]])
    print("DRILL-DONE", flush=True)


if __name__ == "__main__":
    main()
