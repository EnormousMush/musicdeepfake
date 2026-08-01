"""
EnCodec 重建探针(流匹配线 · 第二阶,Batch 11)— 换空间再赌一次 real-only。

第一阶(one_class + typicality)判决:冻结特征空间里 real-only 四种几何全灭,
死因 = "全方位收缩"(假货比人类平滑、干净、少散乱,连小方差方向的能量都偏低)。
本阶换到波形/码本空间:每段音频过 EnCodec 编码→解码,量"复述你时丢了什么"。

预注册(2026-08-01,见 vault 重建探针档案,开跑前写死):
  1. 主假设:假货全方位收缩 ⇒ 更好压 ⇒ 重建误差偏低。分数 = -误差(误差低 = 假),
     表内 EER <50% 即方向命中,>50% 即再次反转(失败判据之一);
  2. 血统格:MusicGen 的音频本来就是 EnCodec 解码产物,预测其重建误差全场最低、
     被抓最准——白送编码器共振假说一格;
  3. 码率梯度:低码率(1.5/3k)应比高码率区分度大(码率越紧,压缩难度差异越暴露)。

两个误差度量 × 5 档码率;码率轴替代第一阶的层数轴,选码率协议照旧
(suno-val vs fma-val),同测试床与判别式/单类逐行对表。
分数缓存到 <data-dir>/recon_scores/encodec.csv,断点续跑。

Usage(服务器 GPU,音频已在位):
  python diagnostics/encodec_recon.py --data-dir data_store/crossgen_export
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eer import compute_eer

SR = 24000
BANDWIDTHS = (1.5, 3.0, 6.0, 12.0, 24.0)
METRICS = ("l1", "mel")


def mel_l1(a, b):
    def logmel(x):
        m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, n_mels=64)
        return np.log(m + 1e-5)
    A, B = logmel(a), logmel(b)
    T = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:, :T] - B[:, :T]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    out_dir = data_dir / "recon_scores"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "encodec.csv"

    cols = ["audio_id"] + [f"{m}_{bw}" for bw in BANDWIDTHS for m in METRICS]
    done = {}
    if out_csv.exists():
        for r in csv.DictReader(open(out_csv)):
            done[r["audio_id"]] = r
    print(f"{len(rows)} rows in manifest, {len(done)} already scored", flush=True)

    todo = [r for r in rows if r["audio_id"] not in done]
    if todo:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoProcessor, EncodecModel
        proc = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device).eval()

        write_header = not out_csv.exists()
        f_out = open(out_csv, "a", newline="")
        w = csv.DictWriter(f_out, fieldnames=cols)
        if write_header:
            w.writeheader()
        t0, n_fail = time.time(), 0
        with torch.no_grad():
            for i, r in enumerate(todo, 1):
                try:
                    wav, srate = sf.read(data_dir / r["rel_path"], dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    if srate != SR:
                        wav = librosa.resample(wav, orig_sr=srate, target_sr=SR)
                    inputs = proc(raw_audio=wav, sampling_rate=SR, return_tensors="pt")
                    iv = inputs["input_values"].to(device)
                    rec = {"audio_id": r["audio_id"]}
                    for bw in BANDWIDTHS:
                        out = model(iv, bandwidth=bw).audio_values[0, 0].cpu().numpy()
                        T = min(len(wav), len(out))
                        rec[f"l1_{bw}"] = float(np.abs(wav[:T] - out[:T]).mean())
                        rec[f"mel_{bw}"] = mel_l1(wav[:T], out[:T])
                    w.writerow(rec)
                    done[r["audio_id"]] = rec
                except Exception as e:
                    n_fail += 1
                    print(f"  FAIL {r['audio_id']}: {e!r}", flush=True)
                if i % 200 == 0:
                    f_out.flush()
                    print(f"  {i}/{len(todo)} ({time.time()-t0:.0f}s, {n_fail} failed)", flush=True)
        f_out.close()
        print(f"scoring done ({n_fail} failed)", flush=True)

    # ---------- 判读表 ----------
    src = {r["audio_id"]: r["source"] for r in rows}
    sp = {r["audio_id"]: r["split"] for r in rows}
    ids = [i for i in done]
    gens = sorted({src[i] for i in ids} - {"fma"})

    def score_arr(mask_fn, key):
        # 预注册方向:分数 = -误差(误差低 = 假)
        return np.array([-float(done[i][key]) for i in ids if mask_fn(i)])

    def pair_eer(s_real, s_fake):
        yy = np.concatenate([np.zeros(len(s_real)), np.ones(len(s_fake))])
        ss = np.concatenate([s_real, s_fake])
        return compute_eer(yy, ss)["eer"]

    for m in METRICS:
        keys = [f"{m}_{bw}" for bw in BANDWIDTHS]
        val_eers = []
        for k in keys:
            e = pair_eer(score_arr(lambda i: src[i] == "fma" and sp[i] == "val", k),
                         score_arr(lambda i: src[i] == "suno" and sp[i] == "val", k))
            val_eers.append(e)
        bstar = int(np.nanargmin(val_eers))
        print(f"\n=== 重建[{m}] 选码率(suno-val vs fma-val,分数=-误差)===")
        print("kbps       val(suno)")
        for bw, e in zip(BANDWIDTHS, val_eers):
            print(f"{bw:>5} {e*100:>10.2f}%")
        print(f"最佳码率(val):B* = {BANDWIDTHS[bstar]} kbps")
        print(f"\n=== 重建[{m}] 跨生成器 EER(每生成器 vs fma-test)===")
        print(f"{'generator':>20} {'@B*':>9} {'best-bw':>18}")
        for g in gens:
            row = [pair_eer(score_arr(lambda i: src[i] == "fma" and sp[i] == "test", k),
                            score_arr(lambda i: src[i] == g, k)) for k in keys]
            bb = int(np.nanargmin(row))
            print(f"{g:>20} {row[bstar]*100:>8.2f}% "
                  f"{row[bb]*100:>10.2f}% ({BANDWIDTHS[bb]}k)")

    print("\n判读:<50% = 预注册方向命中('假货更好压');>50% = 再次反转(失败判据)。"
          "预测 MusicGen 全场最低(EnCodec 血统格);低码率应更锋利。"
          "与判别式/单类同测试床逐行对表。", flush=True)


if __name__ == "__main__":
    main()
