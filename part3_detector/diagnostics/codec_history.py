"""
编码历史检查 — 两类都过同一个 mp3(默认 128kbps)再解回,重抽特征、逐层重探针,与原始对比。

回答:探针吃的是不是"编码链差异"(Suno 平台 mp3 vs FMA 各年代 mp3 的压缩痕迹)?
  - 重编码后 EER 大幅上升 -> 之前主要吃编码历史差(统一重编码要进标准预处理)
  - 基本不动             -> 信号在压缩痕迹之下,编码历史从嫌疑名单划掉

mp3 往返优先用 ffmpeg(启动时自检);没有 ffmpeg 就退回 lameenc + soundfile
(pip install lameenc,清华源可装;soundfile 需 mp3 解码支持,同样启动自检)。
特征缓存到 features_mp3{bitrate}/,重跑秒级恢复。

Usage (server, venv active, GPU1; MERT 6000 clips 约 1-2h):
  python diagnostics/codec_history.py --data-dir data_store/subset_export_round1 \
      --encoder mert --bitrate 128
  加 --limit 20 先干跑验证 mp3 往返能动。
"""
import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifiers import linear as linear_clf
from eval.eer import compute_eer
from diagnostics.bandwidth_ablation import probe_all_layers, load_cached


def roundtrip_ffmpeg(src_path, bitrate_k, tmpdir):
    """flac -> mp3(bitrate) -> wav,返回 (wav float32, sr)。"""
    mp3 = os.path.join(tmpdir, "t.mp3")
    wav = os.path.join(tmpdir, "t.wav")
    # -nostdin + stdin=DEVNULL:防止 ffmpeg 把终端按键当交互命令(会卡在 "Enter command:")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src_path),
                    "-codec:a", "libmp3lame", "-b:a", f"{bitrate_k}k", mp3],
                   check=True, stdin=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", mp3, wav],
                   check=True, stdin=subprocess.DEVNULL)
    x, sr = sf.read(wav)
    return np.asarray(x, dtype=np.float32), sr


def roundtrip_lameenc(src_path, bitrate_k, _tmpdir):
    """无 ffmpeg 的退路:lameenc 编 mp3,soundfile 解(需 libsndfile>=1.1)。"""
    import lameenc
    x, sr = sf.read(src_path)
    pcm = (np.clip(np.asarray(x, dtype=np.float64), -1, 1) * 32767).astype(np.int16)
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate_k); enc.set_in_sample_rate(sr)
    enc.set_channels(1); enc.set_quality(2)
    data = enc.encode(pcm.tobytes()) + enc.flush()
    y, sr2 = sf.read(io.BytesIO(bytes(data)))
    return np.asarray(y, dtype=np.float32), sr2


def pick_backend():
    if shutil.which("ffmpeg"):
        print("mp3 backend: ffmpeg")
        return roundtrip_ffmpeg
    print("ffmpeg 不在 -> 试 lameenc + soundfile ...")
    try:
        import lameenc  # noqa: F401
        probe = np.sin(np.linspace(0, 2 * np.pi * 440, 24000)).astype(np.float64)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "probe.wav")
            sf.write(p, probe, 24000)
            y, _ = roundtrip_lameenc(p, 128, td)
        assert len(y) > 1000
        print("mp3 backend: lameenc + soundfile(自检通过)")
        return roundtrip_lameenc
    except Exception as e:
        sys.exit(f"两个 mp3 后端都不可用:{e!r}\n"
                 f"  修法:pip install lameenc -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                 f"  若 soundfile 不支持 mp3 解码:pip install -U soundfile(同上源)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--encoder", default="mert")
    ap.add_argument("--bitrate", type=int, default=128, help="kbps,两类同值")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from encoders.ssl import SSLEncoder

    rt = pick_backend()
    data_dir = Path(args.data_dir)
    rows = list(csv.DictReader(open(data_dir / "manifest.csv")))
    if args.limit:
        keep, seen = [], {0: 0, 1: 0}
        for r in rows:
            lb = int(r["label"])
            if seen[lb] < args.limit:
                keep.append(r); seen[lb] += 1
        rows = keep
        print(f"Dry-run: {len(rows)} clips")

    enc = SSLEncoder(args.encoder, device=args.device)
    print(f"Encoder: {args.encoder} on {enc.device} | mp3 {args.bitrate}kbps 往返(两类一视同仁)")

    cache = data_dir / f"features_mp3{args.bitrate}" / args.encoder
    cache.mkdir(parents=True, exist_ok=True)

    F, y, sp, failures = [], [], [], []
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=str(data_dir)) as tmpdir:
        for i, r in enumerate(rows, 1):
            cpath = cache / f"{r['audio_id']}.npy"
            try:
                if cpath.exists():
                    f = np.load(cpath)
                else:
                    wav, srate = rt(data_dir / r["rel_path"], args.bitrate, tmpdir)
                    f = enc.encode_all_layers(wav, srate)
                    np.save(cpath, f)
                F.append(f); y.append(int(r["label"])); sp.append(r["split"])
            except Exception as e:
                failures.append((r["audio_id"], repr(e)))
            if i % 100 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s, {len(failures)} failed)")

    F = np.stack(F); y = np.array(y); sp = np.array(sp)
    rtp = probe_all_layers(F, y, sp)

    Fo, yo, spo = load_cached(rows, data_dir / "features" / args.encoder)
    orig = probe_all_layers(Fo, yo, spo) if Fo is not None else None

    print(f"\n=== 逐层 EER:原始 vs mp3 {args.bitrate}k 往返 ===")
    print(f"{'layer':>5} {'orig val':>9} {'orig test':>10} {'mp3 val':>9} {'mp3 test':>10}")
    for L, ev, et in rtp:
        if orig:
            _, oev, oet = orig[L]
            print(f"{L:>5} {oev*100:>8.2f}% {oet*100:>9.2f}% {ev*100:>8.2f}% {et*100:>9.2f}%")
        else:
            print(f"{L:>5} {'--':>9} {'--':>10} {ev*100:>8.2f}% {et*100:>9.2f}%")

    best = min(rtp, key=lambda p: p[1] if np.isfinite(p[1]) else 9)
    print(f"\n重编码后最佳层(val):layer {best[0]}  val {best[1]*100:.2f}%  test {best[2]*100:.2f}%")
    print("判读:mp3 test 相比 orig test 大幅上升 -> 之前吃的是编码历史;基本不动 -> 编码历史排除。")
    if failures:
        print(f"{len(failures)} failed (first few): {failures[:3]}")


if __name__ == "__main__":
    main()
