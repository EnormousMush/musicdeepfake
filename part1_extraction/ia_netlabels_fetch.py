"""
Internet Archive netlabels 第三来源采集(2026-08-16,held-out 拼盘第二源)。

背景:ccMixter 2020+ 池子扫穿只有 ~279–400 首,用户定目标合计 ≥800。
候选调研:Audius 出局(license 全是 All rights reserved),Musopen 出局(IA 镜像仅
34 条目,自家 API 关闭),IA netlabels 当选:2020+ 器乐系标签 2445 个发行,
CC 传统 + 逐条 licenseurl,creator/date 齐全。

野味清洗四层:
  1. 检索层:collection:netlabels + 2020+ + 器乐系 genre 标签(ambient/idm/techno/
     downtempo/instrumental/electronica/drone);
  2. 条目层:标题/标签含 podcast|radio|episode|broadcast|dj set|mixtape 的跳过;
     licenseurl 必须含 creativecommons;
  3. 曲目层:mp3、时长 60–600s(排掉电台长流和碎片),每个发行最多取 2 首;
  4. 全局层:单 creator 上限(默认 5),与 ccMixter 同一稳健性协议。

工程注记(与 ccmixter_fetch 同门):API 走 curl 子进程(本机代理会把大响应搞成
>64KB 单行,python http 库必炸);下载走 requests 流式(定长响应无此病)。

Usage(Mac,Seagate 挂载):
  python3 part1_extraction/ia_netlabels_fetch.py \
    --out "/Volumes/Seagate /honors_paper/1_corpora_real/ia_netlabels" --target 450
  加 --limit 5 干跑(只考察前 5 个发行)。
"""
import argparse
import csv
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

SEARCH = "https://archive.org/advancedsearch.php"
GENRES = '(ambient OR idm OR techno OR downtempo OR instrumental OR electronica OR drone)'
QUERY = f'collection:netlabels AND mediatype:audio AND date:[2020-01-01 TO 2026-12-31] AND subject:{GENRES}'
ROWS = 50
BAD = re.compile(r"podcast|radio|episode|broadcast|dj.?set|mixtape|live.?show", re.I)
UA = {"User-Agent": "honors-thesis-corpus/1.0 (academic research)"}


def curl_json(url, retries=4):
    for i in range(retries):
        try:
            out = subprocess.run(["curl", "-sf", "--max-time", "60", "-A", UA["User-Agent"], url],
                                 capture_output=True, timeout=90).stdout
            return json.loads(out.decode("utf-8", "replace"))
        except Exception as e:
            print(f"  api 第{i+1}次失败: {repr(e)[:60]}", flush=True)
            time.sleep(2 * (i + 1))
    return None


def search_page(page):
    url = (f"{SEARCH}?q={quote(QUERY)}&fl%5B%5D=identifier&fl%5B%5D=creator"
           f"&fl%5B%5D=title&fl%5B%5D=subject&fl%5B%5D=date&fl%5B%5D=licenseurl"
           f"&sort%5B%5D=date+desc&rows={ROWS}&page={page}&output=json")
    d = curl_json(url)
    return (d or {}).get("response", {}).get("docs", [])


def download(url, dst, retries=3):
    for i in range(retries):
        try:
            with requests.get(url, headers=UA, timeout=180, stream=True) as r:
                r.raise_for_status()
                with open(dst, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
            head = open(dst, "rb").read(3)
            if dst.stat().st_size > 300_000 and (head[:3] == b"ID3" or head[0] == 0xFF):
                return True
            dst.unlink(missing_ok=True)
        except Exception as e:
            dst.unlink(missing_ok=True)
            print(f"  下载失败第{i+1}次: {repr(e)[:60]}", flush=True)
        time.sleep(2 * (i + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=450)
    ap.add_argument("--artist-cap", type=int, default=5)
    ap.add_argument("--per-item", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="干跑:只考察前 N 个发行")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest_ia.csv"
    done, per_artist = set(), Counter()
    if man_path.exists():
        for r in csv.DictReader(open(man_path)):
            done.add(r["audio_id"]); per_artist[r["artist"]] += 1
        print(f"resume: 已有 {len(done)} 首", flush=True)
    new_file = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.DictWriter(mf, fieldnames=["audio_id", "item", "artist", "year", "date",
                                       "license", "title", "src_file", "rel_path"])
    if new_file:
        w.writeheader()

    n_have, page, seen_items, t0 = len(done), 1, 0, time.time()
    while n_have < args.target:
        docs = search_page(page)
        if not docs:
            print("检索尽头/失败,停止", flush=True)
            break
        page += 1
        for doc in docs:
            seen_items += 1
            if args.limit and seen_items > args.limit:
                docs = []; break
            ident = doc["identifier"]
            lic = doc.get("licenseurl", "") or ""
            if isinstance(lic, list):
                lic = lic[0] if lic else ""
            creator = doc.get("creator", "?")
            if isinstance(creator, list):
                creator = creator[0] if creator else "?"
            blob = " ".join([ident, str(doc.get("title", "")), str(doc.get("subject", ""))])
            if "creativecommons" not in lic or BAD.search(blob) or per_artist[creator] >= args.artist_cap:
                continue
            meta = curl_json(f"https://archive.org/metadata/{ident}")
            if not meta:
                continue
            year = (doc.get("date") or "")[:4]
            taken = 0
            for f in meta.get("files", []):
                if taken >= args.per_item or per_artist[creator] >= args.artist_cap:
                    break
                if "MP3" not in (f.get("format") or ""):
                    continue
                try:
                    dur = float(f.get("length") or 0)
                except ValueError:   # 有的 length 是 "mm:ss"
                    p = str(f["length"]).split(":")
                    dur = int(p[0]) * 60 + float(p[1]) if len(p) == 2 else 0
                if not (60 <= dur <= 600):
                    continue
                aid = f"ia_{ident}_{re.sub(r'[^A-Za-z0-9]+', '', f['name'])[:40]}"
                if aid in done:
                    taken += 1   # 已采过也计入本发行配额,避免 resume 后超采
                    continue
                url = f"https://archive.org/download/{ident}/{quote(f['name'])}"
                dst = out / "audio" / f"{aid}.mp3"
                if not download(url, dst):
                    continue
                w.writerow(dict(audio_id=aid, item=ident, artist=creator, year=year,
                                date=doc.get("date", ""), license=lic,
                                title=str(doc.get("title", ""))[:80], src_file=f["name"],
                                rel_path=f"audio/{aid}.mp3"))
                mf.flush()
                done.add(aid); per_artist[creator] += 1; n_have += 1; taken += 1
                if n_have % 25 == 0:
                    print(f"  {n_have}/{args.target} ({time.time()-t0:.0f}s, "
                          f"创作者 {len(per_artist)}, 已扫发行 {seen_items})", flush=True)
                if n_have >= args.target:
                    break
                time.sleep(0.3)
            if n_have >= args.target:
                docs = []; break
        if args.limit and seen_items > args.limit:
            break
        if not docs and page > 2:
            break
    yrs = Counter(r["year"] for r in csv.DictReader(open(man_path)))
    print(f"done: {n_have} 首, {len(per_artist)} 创作者 -> {man_path}")
    print("年份分布: " + " ".join(f"{y}:{c}" for y, c in sorted(yrs.items())), flush=True)


if __name__ == "__main__":
    main()
