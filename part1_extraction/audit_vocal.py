"""
器乐验证:用 demucs 验 Jamendo `musicinfo.vocalinstrumental` 字段准不准(2026-09-19)。

jamendo_sweep 的 20,980 首**全部**靠这个元数据字段定性为器乐,我们一个音频都没听过。
这个字段准不准,直接决定这批数据能不能按现在的口径写进论文。

方法:demucs 分离出 vocals 轨,算它占全曲能量的比例。真器乐的 vocals 轨只有
分离残差(能量极低);有人声的会明显高。**用已知人声/已知器乐的参照集定阈值**,
而不是拍一个 —— 否则阈值本身就是个没根据的自由参数。

Usage:
  .venv_audit/bin/python part1_extraction/audit_vocal.py --audit-dir SCRATCH/audit --n 100
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")


def load_demucs(device):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model = get_model("htdemucs").to(device).eval()
    return model, apply_model


@torch.no_grad()
def vocal_ratio(path, model, apply_model, device):
    """返回 vocals 轨能量占比(0–1)。分离失败返回 None。"""
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y])          # demucs 要立体声
    else:
        y = y.T
    if sr != model.samplerate:
        import librosa
        y = np.stack([librosa.resample(c, orig_sr=sr, target_sr=model.samplerate)
                      for c in y])
    wav = torch.from_numpy(y).unsqueeze(0).to(device)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    try:
        est = apply_model(model, wav, device=device, split=True, overlap=0.1)[0]
    except Exception as exc:
        print(f"    分离失败 {Path(path).name}: {type(exc).__name__}: {exc}", flush=True)
        return None
    srcs = model.sources                      # ['drums','bass','other','vocals']
    e = {s: float((est[i] ** 2).sum()) for i, s in enumerate(srcs)}
    tot = sum(e.values()) + 1e-12
    return e.get("vocals", 0.0) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--n", type=int, default=100, help="每个待验层抽多少首")
    ap.add_argument("--n-ref", type=int, default=40, help="每个参照组抽多少首(定阈值用)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    d = Path(args.audit_dir)
    rows = list(csv.DictReader(open(d / "index.csv", encoding="utf-8")))
    # htdemucs 在 MPS 上会撞 "Output channels > 65536 not supported",只能走 CPU/CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.threads)
    print(f"设备 {device} (MPS 不支持 htdemucs 的宽卷积), 线程 {args.threads}")
    model, apply_model = load_demucs(device)
    print(f"htdemucs 已载入,stems={model.sources}  sr={model.samplerate}")

    # 待验 + 参照(参照用来定阈值,不能拍脑袋)
    pick = []
    for g, n in [("acestep_self_core", args.n),
                 ("ref_human", args.n_ref), ("ref_ai", args.n_ref)]:
        sub = [r for r in rows if r["group"] == g][:n]
        pick += sub
        print(f"  {g}: {len(sub)} 首")

    out, fail = [], 0
    for i, r in enumerate(pick, 1):
        v = vocal_ratio(Path(r["clip"]), model, apply_model, device)
        if v is None:
            fail += 1
            continue
        out.append(dict(audio_id=r["audio_id"], group=r["group"],
                        claimed=r.get("claimed", ""), year=r.get("year", ""),
                        vocal_ratio=round(v, 5)))
        if i % 20 == 0:
            print(f"  {i}/{len(pick)}", flush=True)

    if not out:
        print("⚠️ 没有一首分离成功,见上面的失败原因")
        return
    with open(d / "vocal_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"\n{'='*60}\nvocals 轨能量占比(越高 = 越可能有人声),失败 {fail}\n{'='*60}")
    summ = {}
    for g in ["ref_human", "acestep_self_core"]:
        v = np.array([r["vocal_ratio"] for r in out if r["group"] == g])
        if not len(v):
            continue
        q = np.percentile(v, [10, 50, 90, 95])
        print(f"{g:16s} n={len(v):3d}  p10 {q[0]:.4f}  中位 {q[1]:.4f} "
              f" p90 {q[2]:.4f}  p95 {q[3]:.4f}  max {v.max():.4f}")
        summ[g] = dict(n=len(v), median=float(q[1]), p90=float(q[2]),
                       max=float(v.max()))

    # 参照人类里有已知器乐白名单(ref_jam_*),拿它的 p95 当"器乐上界"
    ref_inst = np.array([r["vocal_ratio"] for r in out
                         if r["audio_id"].startswith("ref_jam_")])
    if len(ref_inst) >= 10:
        thr = float(np.percentile(ref_inst, 95))
        print(f"\n阈值 = 已知器乐参照(jamendo2025 白名单 n={len(ref_inst)})的 p95 = {thr:.4f}")
        for g in ["acestep_self_core"]:
            v = np.array([r["vocal_ratio"] for r in out if r["group"] == g])
            if len(v):
                bad = int((v > thr).sum())
                print(f"  {g:16s} 超阈值 {bad}/{len(v)} = {bad/len(v):.1%}  "
                      f"→ 字段准确率约 {1-bad/len(v):.1%}")
                summ.setdefault(g, {})["over_thr_pct"] = float(bad / len(v))
        summ["threshold"] = thr

    (d / "vocal_audit_report.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细 → {d/'vocal_audit.csv'}\n报告 → {d/'vocal_audit_report.json'}")


if __name__ == "__main__":
    main()
