"""
SONICS devocal 附表(Batch 7b)· 共同规格导出 + manifest 组装。

产出 devocal_export 包,直接喂 run_crossgen(零代码改动):
- 训练/验证行:从主库 manifest 复制 suno/fma 的 train/val 行(rel_path 不变,
  服务器端把 audio/suno、audio/fma 软链到 crossgen_export 即可,不重复传数据);
- 测试行(全部 demucs 伴奏轨 -> 10s/16k 共同规格,音频放 audio/dv/):
  fma_ref(label 0)/ suno_ref(label 1)= 域内桥行;5 个 SONICS 版本 = 生成器行。
探针在无刀数据上训练、在有刀对上评测——两个测试类共享处理历史,
suno_ref 行给出 demucs 的信号损耗汇率。

  python devocal_export.py --stems "/Volumes/Seagate /sonics/stems" \
      --main-manifest "/Volumes/Seagate /frank-suno-round1/crossgen_export/manifest.csv" \
      --out "/Volumes/Seagate /frank-suno-round1/devocal_export"
"""
import argparse
import csv
import time
from pathlib import Path

from crossgen_prep import process_one

GROUPS = {  # 目录名 -> (manifest source, label)
    "sunov2":   ("sunov2_dv", 1),
    "sunov3":   ("sunov3_dv", 1),
    "sunov35":  ("sunov35_dv", 1),
    "udio30":   ("udio30_dv", 1),
    "udio120":  ("udio120_dv", 1),
    "suno_ref": ("suno", 1),
    "fma_ref":  ("fma", 0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, help="sonics/stems 根目录")
    ap.add_argument("--main-manifest", required=True, help="主库 manifest(取 train/val 行)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=float, default=10.0, help="30s 段取 10-20s 窗")
    ap.add_argument("--limit", type=int, default=None, help="干跑:每组 N 个")
    args = ap.parse_args()

    stems = Path(args.stems)
    out = Path(args.out); (out / "audio" / "dv").mkdir(parents=True, exist_ok=True)
    rows_out, failures, t0 = [], [], time.time()

    # ---- 1) 训练/验证行:主库 suno/fma 原样搬运 ----
    main_rows = list(csv.DictReader(open(args.main_manifest)))
    for r in main_rows:
        if r["source"] in ("suno", "fma") and r["split"] in ("train", "val"):
            rows_out.append(dict(audio_id=r["audio_id"], source=r["source"],
                                 label=r["label"], split=r["split"],
                                 rel_path=r["rel_path"]))
    print(f"训练/验证行(沿用主库,需服务器软链): {len(rows_out)}")

    # ---- 2) devocal 测试行:伴奏轨 -> 共同规格 ----
    for gdir, (source, label) in GROUPS.items():
        files = sorted((stems / gdir / "htdemucs").glob("*/no_vocals.mp3"))
        if args.limit:
            files = files[: args.limit]
        print(f"  {gdir}: {len(files)} -> source={source}", flush=True)
        for i, p in enumerate(files, 1):
            aid = f"{p.parent.name}_dv"
            rel = f"audio/dv/{aid}.flac"
            try:
                if not (out / rel).exists():
                    process_one(p, out / rel, offset_s=args.offset)
                rows_out.append(dict(audio_id=aid, source=source, label=label,
                                     split="test", rel_path=rel))
            except Exception as e:
                failures.append((aid, repr(e)))
            if i % 500 == 0 or i == len(files):
                print(f"    {i}/{len(files)} ({time.time()-t0:.0f}s, {len(failures)} failed)",
                      flush=True)

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_id", "source", "label", "split", "rel_path"])
        w.writeheader(); w.writerows(rows_out)
    by = {}
    for r in rows_out:
        by[(r["source"], r["split"])] = by.get((r["source"], r["split"]), 0) + 1
    print(f"\n完成:{len(rows_out)} 行 -> {out}/manifest.csv({len(failures)} failed)")
    print("分布:", by)
    if failures:
        print("失败样例:", failures[:5])


if __name__ == "__main__":
    main()
