"""从 final_index.csv 的 core 层产出 SpecTTTra 要的 train/valid/test CSV(filepath,target)。
target: 0=human 1=AI。切分:suno 按 uuid 分组(两个 take 不拆),其余文件级;70/15/15,seed 0。"""
import csv, random, collections, sys
from pathlib import Path
idx, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader(open(idx, encoding="utf-8")) if r["layer"] == "core"]
groups = collections.defaultdict(list)
for r in rows: groups[r["split_key"]].append(r)
keys = sorted(groups); random.Random(0).shuffle(keys)
n = len(keys); cut = {"train": keys[:int(.7*n)], "valid": keys[int(.7*n):int(.85*n)], "test": keys[int(.85*n):]}
for split, ks in cut.items():
    with open(out / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filepath", "target", "corpus", "family", "split_key"])
        c = collections.Counter()
        for k in ks:
            for r in groups[k]:
                w.writerow([r["path"], 1 if r["side"] == "ai" else 0, r["corpus"], r["family"], k]); c[r["side"]] += 1
    print(f"{split:5s} {dict(c)}")
