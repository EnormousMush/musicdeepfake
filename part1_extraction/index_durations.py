"""给 final_index.csv 补时长(秒)。mutagen 只读文件头,mp3/wav/flac 都毫秒级,不解码。"""
import csv, sys, concurrent.futures as cf
from pathlib import Path
from mutagen import File as MF
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, encoding="utf-8")))
def dur(p):
    try:
        m = MF(p); return round(m.info.length, 2) if m and m.info else ""
    except Exception: return ""
with cf.ThreadPoolExecutor(16) as ex:
    ds = list(ex.map(dur, [r["path"] for r in rows]))
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["uid", "side", "corpus", "layer", "duration_s"])
    for r, d in zip(rows, ds): w.writerow([r["uid"], r["side"], r["corpus"], r["layer"], d])
print(f"done {len(rows)} rows, missing {sum(1 for d in ds if d=='')}")
