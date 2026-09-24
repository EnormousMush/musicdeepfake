"""补 mutagen 读不到的时长:float wav / 部分 flac / 个别 mp3。soundfile.info 读头,mp3 退回 librosa。"""
import csv, sys, concurrent.futures as cf
import soundfile as sf, librosa
src, idx, dst = sys.argv[1], sys.argv[2], sys.argv[3]
paths = {r["uid"]: r["path"] for r in csv.DictReader(open(idx, encoding="utf-8"))}
rows = list(csv.DictReader(open(src, encoding="utf-8")))
def fill(r):
    if r["duration_s"]: return r["duration_s"]
    p = paths.get(r["uid"], "")
    try: return round(sf.info(p).duration, 2)
    except Exception:
        try: return round(librosa.get_duration(path=p), 2)
        except Exception: return ""
with cf.ThreadPoolExecutor(16) as ex: ds = list(ex.map(fill, rows))
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    for r, d in zip(rows, ds): r["duration_s"] = d; w.writerow(r)
print(f"still missing {sum(1 for d in ds if d=='')}")
