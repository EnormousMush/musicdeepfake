"""
ccMixter 第三来源人类语料采集(2026-08-16,人类音乐集混淆 §6 第 7 发:held-out 终审)。

用途:混训修复(fma+jamendo)的泛化审判需要一个两边都没见过的第三人类来源,
**只测不训**。ccMixter = CC 混音社区,与 FMA/Jamendo 生态不重叠,API 有
instrumental 标签与逐条日期/license/歌手元数据。

协议(与 jamendo_fetch 同门):
  - /api/query?f=json&tags=instrumental&sort=date(新→旧),逐页爬;
  - 默认只收 >= --min-year(2020)的上传,爬到日期地板即停;
  - 单歌手上限 --artist-cap(默认 3);目标 --target(默认 800);
  - 断点续采:manifest_ccmixter.csv 已有的 upload_id 跳过;
  - 下载校验:本地文件 > 300KB 且开头为 ID3/MPEG 帧,否则删除重试;
  - 诚实注记:ccMixter 是混音社区,不同曲目可能共享采样/分轨——作为
    held-out 测试集可接受,写论文时在 Limitations 提一句。

Usage(Mac,Seagate 挂载):
  python3 part1_extraction/ccmixter_fetch.py \
    --out "/Volumes/Seagate /honors_paper/1_corpora_real/ccmixter" \
    --target 800
  加 --limit 5 干跑(只走前 5 条候选)。
"""
import argparse
import csv
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import requests

API = "https://ccmixter.org/api/query"
PAGE = 12   # 服务器会悄悄把大 limit 砍到 12;固定 12 并按实际返回条数推进 offset
UA = {"User-Agent": "honors-thesis-corpus/1.0 (academic research)"}
# 翻页用 curl 子进程:本机代理会把大响应搞成 >64KB 的单行,python http 库(_MAXLINE=65536)
# 必炸 LineTooLong;curl 无此限制。偶发慢响应靠重试消化(实测正常 ~2s/页)。
# 下载仍用 requests(定长流式响应,无此病;带 Referer 过防盗链)。


def get_page(offset, retries=5):
    url = f"{API}?f=json&tags=instrumental&sort=date&limit={PAGE}&offset={offset}"
    for i in range(retries):
        try:
            out = subprocess.run(["curl", "-sf", "--max-time", "45", "-A", UA["User-Agent"], url],
                                 capture_output=True, timeout=60).stdout
            # 个别老条目的简介含非 UTF-8 字节(我们不用该字段),容错替换,别让整页报废
            data = json.loads(out.decode("utf-8", "replace"))
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f"  page offset={offset} 第{i+1}次失败: {repr(e)[:60]}", flush=True)
        time.sleep(2 * (i + 1))
    return []


def parse_year(s):
    # "Fri, Aug 14, 2026 @ 3:43 AM"
    m = re.search(r"(\d{4})", s or "")
    return int(m.group(1)) if m else 0


def pick_file(up):
    for f in up.get("files", []):
        info = f.get("file_format_info", {})
        if info.get("media-type") == "audio" and info.get("default-ext") == "mp3" \
                and not f.get("file_is_remote") and f.get("download_url"):
            # 时长预过滤:有的上传 mp3 只是几秒试听(正片是 FLAC),ps 形如 "4:14"
            ps = info.get("ps") or ""
            try:
                mm, ss = ps.split(":"); dur = int(mm) * 60 + int(ss)
            except ValueError:
                dur = None
            if dur is not None and dur < 60:
                return None
            return f["download_url"]
    return None


def download(url, dst, referer, retries=4):
    # 内容服务器有防盗链:必须带 Referer(该曲目的 file_page_url),否则 403。
    # 下载走 https 没问题(炸的只是 >64KB 的 API 响应头,不是流式 body)。
    hdr = dict(UA, Referer=referer or "https://ccmixter.org/")
    for i in range(retries):
        try:
            with requests.get(url, headers=hdr, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(dst, "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
            head = open(dst, "rb").read(3)
            if dst.stat().st_size > 300_000 and (head[:3] == b"ID3" or head[0] == 0xFF):
                return True
            dst.unlink(missing_ok=True)
            print(f"  校验失败(太小/非mp3),重试 {dst.name}", flush=True)
        except Exception as e:
            dst.unlink(missing_ok=True)
            print(f"  下载失败第{i+1}次: {repr(e)[:60]}", flush=True)
        time.sleep(2 * (i + 1))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=800)
    ap.add_argument("--min-year", type=int, default=2020)
    ap.add_argument("--artist-cap", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="干跑:只考察前 N 条候选")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest_ccmixter.csv"
    done, per_artist = set(), Counter()
    if man_path.exists():
        for r in csv.DictReader(open(man_path)):
            done.add(r["upload_id"]); per_artist[r["artist"]] += 1
        print(f"resume: 已有 {len(done)} 首", flush=True)
    new_file = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.DictWriter(mf, fieldnames=["audio_id", "upload_id", "artist", "year",
                                       "date", "license", "url", "rel_path"])
    if new_file:
        w.writeheader()

    n_have, offset, seen_cand, t0 = len(done), 0, 0, time.time()
    while n_have < args.target:
        page = get_page(offset)
        if not page:
            print("翻页尽头/连续失败,停止", flush=True)
            break
        offset += len(page)   # 按实际条数推进:服务器可能给少于 PAGE 条,按 PAGE 推会漏采
        stop = False
        for up in page:
            seen_cand += 1
            if args.limit and seen_cand > args.limit:
                stop = True; break
            year = parse_year(up.get("upload_date_format"))
            if year and year < args.min_year:
                print(f"到达日期地板 {year} < {args.min_year},停止", flush=True)
                stop = True; break
            uid = str(up["upload_id"]); artist = up.get("user_name", "?")
            if uid in done or per_artist[artist] >= args.artist_cap:
                continue
            url = pick_file(up)
            if not url:
                continue
            aid = f"cc_{uid}"
            dst = out / "audio" / f"{aid}.mp3"
            if not download(url, dst, up.get("file_page_url")):
                continue
            w.writerow(dict(audio_id=aid, upload_id=uid, artist=artist, year=year,
                            date=up.get("upload_date_format", ""),
                            license=up.get("license_name", ""), url=url,
                            rel_path=f"audio/{aid}.mp3"))
            mf.flush()
            done.add(uid); per_artist[artist] += 1; n_have += 1
            if n_have % 25 == 0:
                print(f"  {n_have}/{args.target} ({time.time()-t0:.0f}s, "
                      f"歌手 {len(per_artist)})", flush=True)
            if n_have >= args.target:
                stop = True; break
            time.sleep(0.3)
        if stop:
            break
    yrs = Counter()
    for r in csv.DictReader(open(man_path)):
        yrs[r["year"]] += 1
    print(f"done: {n_have} 首, {len(per_artist)} 歌手 -> {man_path}")
    print("年份分布: " + " ".join(f"{y}:{c}" for y, c in sorted(yrs.items())), flush=True)


if __name__ == "__main__":
    main()
