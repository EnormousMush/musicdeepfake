"""
AI 污染审计:拿自家鉴伪线给新语料打分(2026-09-19)。

**这不是"证明不是 AI"**,是异常检测。方法:
  1. 冻结 MERT 提特征(已知人类参照 vs 已知 AI 参照 vs 待验曲目)
  2. 在两组**参照**上训线性头,留出集报 EER——先确认这把尺子本身准不准
  3. 用这把尺子给待验曲目打分,看**分布**:
     - 待验分布贴着人类参照 → 没有系统性污染迹象
     - 冒出一簇贴着 AI 参照的高分 → 实锤嫌疑,拉出来人工听

局限(必须写进结论):
  * 尺子只见过我们这几家生成器,**没见过的家族可能整体漏检**(这正是 P2 的论点);
  * 低分不等于"是人做的",只等于"不像我们见过的 AI";
  * 若这批将来当训练数据,拿模型给自己训练集打分有循环论证——只能当审计,不能当证据。

Usage:
  .venv_audit/bin/python part1_extraction/audit_ai.py --audit-dir SCRATCH/audit
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

MERT_ID = "m-a-p/MERT-v1-95M"
MERT_SR = 24000


def load_encoder(device):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True).to(device).eval()
    return proc, model


@torch.no_grad()
def embed(paths, proc, model, device, batch=8):
    """每片 → [时间均值 ‖ 时间标准差] 池化。

    MERT 的 remote code 在 transformers 5.x 下不返回 hidden_states(只给
    last_hidden_state),所以不做多层拼接。补上**时间维标准差**:我们自己的
    实验反复量到「时间波动小 = AI」是强信号(mfcc_stds / centroid_std /
    rms_std 那一族),均值池化恰好把这个信息池掉了。
    """
    import librosa
    out = []
    for i in range(0, len(paths), batch):
        waves = []
        for p in paths[i:i + batch]:
            y, sr = sf.read(str(p), dtype="float32")
            if sr != MERT_SR:                      # 切片是 16k,MERT 要 24k
                y = librosa.resample(y, orig_sr=sr, target_sr=MERT_SR)
            waves.append(y)
        inp = proc(waves, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        h = model(**inp).last_hidden_state          # (B, T, D)
        feat = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1)
        out.append(feat.float().cpu().numpy())
        if (i // batch) % 10 == 0:
            print(f"    {min(i+batch, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out, 0)


def eer(pos, neg):
    """等错误率 + 对应阈值。pos=AI 分数,neg=人类分数。"""
    cands = np.unique(np.concatenate([pos, neg]))
    bd, best, bt = np.inf, 1.0, 0.0
    for t in cands:
        far = float((neg >= t).mean())          # 人类被判成 AI
        frr = float((pos < t).mean())           # AI 被漏掉
        if abs(far - frr) < bd:
            bd, best, bt = abs(far - frr), (far + frr) / 2, float(t)
    return best, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    d = Path(args.audit_dir)
    rows = list(csv.DictReader(open(d / "index.csv", encoding="utf-8")))
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}   片段 {len(rows)}")

    cache = d / "mert_feats.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        X, ids = z["X"], list(z["ids"])
        print(f"[缓存] 读到 {X.shape}")
    else:
        print("[MERT] 提特征 …")
        proc, model = load_encoder(device)
        paths = [r["clip"] for r in rows]
        X = embed(paths, proc, model, device, args.batch)
        ids = [r["audio_id"] for r in rows]
        np.savez_compressed(cache, X=X, ids=np.array(ids, dtype=object))
        print(f"[MERT] {X.shape} → {cache}")

    by_id = {r["audio_id"]: r for r in rows}
    grp = np.array([by_id[i]["group"] for i in ids])
    Xh, Xa = X[grp == "ref_human"], X[grp == "ref_ai"]
    print(f"\n参照: 人类 {len(Xh)}  AI {len(Xa)}")

    Xr = np.concatenate([Xh, Xa])
    yr = np.concatenate([np.zeros(len(Xh)), np.ones(len(Xa))])
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-8

    # --- 选正则强度:1536 维配几百样本,C 太大必然过拟合
    print("\n[调参] 5 折留出 EER(选 C):")
    best_C, best_e, best_scores = None, 1.0, None
    for C in [0.001, 0.01, 0.1, 1.0]:
        sc = np.zeros(len(Xr))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xr, yr):
            c = LogisticRegression(max_iter=5000, C=C).fit((Xr[tr] - mu) / sd, yr[tr])
            sc[te] = c.predict_proba((Xr[te] - mu) / sd)[:, 1]
        e_, _ = eer(sc[yr == 1], sc[yr == 0])
        print(f"    C={C:<7} EER {e_:.2%}")
        if e_ < best_e:
            best_C, best_e, best_scores = C, e_, sc
    print(f"  → 选 C={best_C},留出 EER {best_e:.2%}")

    # --- 阈值:定在**留出人类**分数的 95 分位 = 5% 误报点。
    #     这样 sweep 的超阈值率就有了直接可比的基线:没有污染时应该也是 ~5%。
    h_out = best_scores[yr == 0]
    thr = float(np.percentile(h_out, 95))
    base_fpr = float((h_out >= thr).mean())
    a_out = best_scores[yr == 1]
    recall = float((a_out >= thr).mean())
    print(f"\n[尺子] 阈值 {thr:.4f}(留出人类 p95)")
    print(f"       该阈值下:人类误报 {base_fpr:.1%}(基线) · 已知 AI 检出 {recall:.1%}")

    # --- 待验曲目:样本外
    clf = LogisticRegression(max_iter=5000, C=best_C).fit((Xr - mu) / sd, yr)
    print(f"\n[打分] 分数越高越像 AI。**参照组用留出分数**,sweep 用样本外分数")
    report = dict(C=best_C, eer=float(best_e), threshold=thr,
                  baseline_fpr=base_fpr, known_ai_recall=recall, groups={})

    def emit(g, s, note=""):
        q = np.percentile(s, [50, 90, 95, 99])
        over = float((s >= thr).mean())
        print(f"  {g:16s} n={len(s):4d}  中位 {q[0]:.3f}  p90 {q[1]:.3f}"
              f"  p99 {q[3]:.3f}   超阈值 {over:6.1%} {note}")
        report["groups"][g] = dict(n=len(s), median=float(q[0]), p90=float(q[1]),
                                   p99=float(q[3]), over_thr=over)
        return over

    emit("ref_human(留出)", h_out, "← 基线,按构造 ≈5%")
    emit("ref_ai(留出)", a_out, "← 已知 AI")
    for g in ["pixabay_quarantine"]:
        m = grp == g
        if not m.any():
            continue
        s = clf.predict_proba((X[m] - mu) / sd)[:, 1]
        over = emit(g, s)
        lift = over / base_fpr if base_fpr > 0 else float("nan")
        print(f"      → 相对基线 {lift:.1f}×"
              + ("  ⚠️ 明显高于基线,值得人工听" if lift >= 2 else
                 "  基线水平,无系统性污染迹象"))
        report["groups"][g]["lift_over_baseline"] = lift
        idx = np.where(m)[0][np.argsort(-s)][:5]
        top = clf.predict_proba((X[idx] - mu) / sd)[:, 1]
        print("      最高分 5 首: " +
              ", ".join(f"{ids[j]}({v:.2f})" for j, v in zip(idx, top)))
        report["groups"][g]["top5"] = [ids[j] for j in idx]

    # 逐曲分数落盘(apply_ai_scores.py 回写索引用)
    with open(d / "ai_scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["audio_id", "group", "score"])
        for g in report["groups"]:
            if g.startswith("ref_"):
                continue
            m = grp == g
            for j, sc in zip(np.where(m)[0], clf.predict_proba((X[m] - mu) / sd)[:, 1]):
                w.writerow([ids[j], g, f"{sc:.6f}"])
    (d / "ai_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 → {d/'ai_audit_report.json'}")


if __name__ == "__main__":
    main()
