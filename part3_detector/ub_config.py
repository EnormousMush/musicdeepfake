"""
生成 U-B(SpecTTTra-α,从头训,只用我们的 core)的训练配置(2026-09-24)。

以官方 spectttra_f1t3 为底,改四处:
  * dataset 指向 prep_ub.py 产出的 16k/定长 wav 与 csv(必须,见 prep_ub.py 头注)
  * audio.max_time = 窗长;model.input_shape 的帧数随之 = ceil(max_time*16000/hop)
  * batch/grad_accum 按本机显存改:官方 bs=96 是 48 GB 卡的设置
  * resume=null,use_init_weights=false → 从头训,不加载官方权重(这就是 U-B 的定义)

Usage:
  python part3_detector/ub_config.py --data DIR --window 10 --bs 32 --accum 3 --epochs 50 \
      --out configs/ub_w10.yaml
"""
import argparse, math, os, yaml
from pathlib import Path

# SpecTTTra 仓库位置(github.com/awsaf49/sonics),默认 ~/Developer/sonics
SONICS_DIR = Path(os.environ.get("SONICS_DIR", Path.home() / "Developer" / "sonics"))
BASE = SONICS_DIR / "configs" / "spectttra_f1t3-5s.yaml"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--window", type=float, required=True)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--accum", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(BASE))
    hop = cfg["melspec"]["hop_length"]
    n_frames = math.ceil(a.window * cfg["audio"]["sample_rate"] / hop) + 1
    cfg["experiment_name"] = a.name or f"ub_spectttra_alpha_w{int(a.window)}"
    cfg["dataset"] = {k: f"{a.data}/{s}.csv" for k, s in
                      [("train_dataframe", "train"), ("valid_dataframe", "valid"), ("test_dataframe", "test")]}
    cfg["audio"]["max_time"] = a.window
    cfg["audio"]["random_sampling"] = False       # 片段已定长,不再随机裁
    cfg["model"]["input_shape"] = [cfg["melspec"]["n_mels"], n_frames]
    cfg["model"]["resume"] = None; cfg["model"]["use_init_weights"] = False
    cfg["training"]["batch_size"] = a.bs; cfg["training"]["epochs"] = a.epochs
    cfg["validation"]["batch_size"] = a.bs
    cfg["optimizer"]["grad_accum_steps"] = a.accum
    cfg["environment"]["num_workers"] = a.workers
    # 官方 lr 是按 global batch 线性缩放的;等效 batch = bs*accum
    cfg["scheduler"]["lr"] = cfg["scheduler"]["lr_base"] * (a.bs * a.accum) / cfg["scheduler"]["lr_base_size"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(a.out, "w"), sort_keys=False)
    print(f"→ {a.out}  window={a.window}s  input_shape={cfg['model']['input_shape']}  "
          f"eff_batch={a.bs*a.accum}  lr={cfg['scheduler']['lr']:.2e}")

if __name__ == "__main__":
    main()
