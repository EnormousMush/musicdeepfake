"""
Jamendo 器乐语料全量扫库(人类数据集扩张,2026-09-18)。

需求只有两条:**尽量不是 AI 做的** + **尽量是器乐**。不做 genre 配额。

为什么分三段而不是边下边存:
    AI 嫌疑里最硬的信号是**艺术家级行为**(一个账号短期内批量上传大量曲目
    = 灌库/AI 农场模式),这是 E13 那次抓出 12 个账号 69 首污染的同一招。
    这个信号必须先看到**全量人口**才能算——边下边判是判不出来的。
    所以:harvest(只拉元数据,不下音频)→ audit(全量打分)→ fetch(只下干净的)。

实测到的 API 事实(2026-09-18):
  * `fuzzytags=instrumental` 池深约 8–12 万首,各深度抽检器乐率 **100%**
  * limit 上限 200;offset 至少能到 10 万,没有低天花板
  * **会间歇性返回空页且 `headers.status` 仍是 success** —— 空页必须重试确认
  * `vocalinstrumental` 不能当顶层过滤参数(传了返回 0),只能逐曲读 musicinfo

前置(https://devportal.jamendo.com 免费注册,**不要提交进 git**):
  export JAMENDO_CLIENT_ID=xxxxxxxx

Usage:
  # 1) 扫元数据(不下音频,几分钟)
  python part1_extraction/jamendo_sweep.py harvest --out DIR --max-tracks 100000

  # 2) 全量 AI 审计 + 器乐复核,打印人口画像
  python part1_extraction/jamendo_sweep.py audit --out DIR

  # 3) 下音频(只下审计通过的)
  python part1_extraction/jamendo_sweep.py fetch --out DIR --target 20000 --workers 8

断点续跑:三段都可断可续。harvest 按 track_id 去重续写;fetch 跳过已校验的 mp3。
授权:逐曲记录 license_ccurl;仅本地研究分析,不再分发。
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import datetime as dt
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

API = "https://api.jamendo.com/v3.0/tracks/"

# 扫库入口:每个 tag 独立翻页,合并去重。instrumental 是主入口,
# 其余是器乐重镇,用来触达主入口 offset 够不到的长尾。
SWEEP_TAGS = [
    "instrumental", "ambient", "classical", "electronic", "jazz", "lounge",
    "soundtrack", "piano", "guitar", "orchestral", "chillout", "newage",
    "postrock", "downtempo", "acoustic", "experimental", "folk", "blues",
]

# 生成器关键词:命中即硬排除(不打分,直接踢)
AI_KEYWORDS = re.compile(
    r"\b(ai[\s\-_]?generated|generated[\s\-_]?by[\s\-_]?ai|ai[\s\-_]?music|"
    r"ai[\s\-_]?composed|suno|udio|mubert|soundraw|aiva|boomy|loudly|beatoven|"
    r"stable[\s\-_]?audio|musicgen|audiocraft|riffusion|text[\s\-_]?to[\s\-_]?music|"
    r"neural[\s\-_]?compos|machine[\s\-_]?generated|prompt[\s\-_]?generated)\b",
    re.I,
)

# Suno 公测上线 = 生成式音乐进入开放平台的分水岭。此前上传结构上不可能是 AI 曲。
AI_ERA_CUTOFF = "2023-01-01"

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")

META = "meta.jsonl"
CLEAN = "clean.csv"
MANIFEST = "manifest.csv"

FIELDS = [
    "audio_id", "track_id", "track_name", "artist_id", "artist_name", "album_id",
    "releasedate", "year", "duration", "vocalinstrumental", "acousticelectric",
    "speed", "mi_genres", "mi_instruments", "mi_vartags", "license_ccurl",
    "ai_risk", "ai_reasons", "audiodownload", "audio_path",
]


def _is_valid_mp3(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 10_000:
        return False
    with open(p, "rb") as f:
        return any(f.read(3)[: len(m)] == m for m in _MP3_MAGIC)


def _get_page(params, tries=6):
    """一页,带重试。空页也重试——Jamendo 空页不报错,不能当翻到底。"""
    for a in range(tries):
        try:
            r = requests.get(API, params=params, timeout=60)
            r.raise_for_status()
            body = r.json()
            if body.get("headers", {}).get("status") != "success":
                raise RuntimeError(body.get("headers"))
            rows = body.get("results", [])
            if rows or a == tries - 1:
                return rows
        except Exception:
            if a == tries - 1:
                raise
        time.sleep(1.2 * (a + 1))
    return []


def flatten(t: dict) -> dict:
    mi = t.get("musicinfo", {}) or {}
    tags = mi.get("tags", {}) or {}
    rd = t.get("releasedate", "") or ""
    return dict(
        track_id=str(t.get("id", "")), track_name=t.get("name", ""),
        artist_id=str(t.get("artist_id", "")), artist_name=t.get("artist_name", ""),
        album_id=str(t.get("album_id", "")), releasedate=rd, year=rd[:4],
        duration=t.get("duration", ""),
        vocalinstrumental=mi.get("vocalinstrumental", ""),
        acousticelectric=mi.get("acousticelectric", ""),
        speed=mi.get("speed", ""),
        mi_genres="|".join(tags.get("genres", []) or []),
        mi_instruments="|".join(tags.get("instruments", []) or []),
        mi_vartags="|".join(tags.get("vartags", []) or []),
        license_ccurl=t.get("license_ccurl", ""),
        audiodownload=t.get("audiodownload", "") if t.get("audiodownload_allowed") else "",
    )


# ------------------------------------------------------------------ harvest

def cmd_harvest(args):
    cid = os.environ["JAMENDO_CLIENT_ID"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta_path = out / META

    seen = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["track_id"])
                except Exception:
                    pass
        print(f"[续跑] 已有 {len(seen)} 条元数据")

    # 面 = (tag) 或 (tag, year)。不切年份时 order=popularity_total 会把老歌顶上来,
    # 近几年被严重欠采样;所以对指定年份再单独切一遍,保证年代覆盖而不是碰运气。
    facets = [(t, None) for t in args.tags]
    for y in args.years:
        facets += [(t, y) for t in args.tags]

    added = 0
    with open(meta_path, "a", encoding="utf-8") as f:
        for tag, year in facets:
            label = tag if year is None else f"{tag}@{year}"
            offset, empty_streak, tag_new = 0, 0, 0
            while len(seen) < args.max_tracks:
                params = dict(
                    client_id=cid, format="json", limit=200, offset=offset,
                    fuzzytags=tag, audioformat="mp32", include="musicinfo licenses",
                    order="popularity_total",
                )
                if year is not None:
                    params["datebetween"] = f"{year}-01-01_{year}-12-31"
                rows = _get_page(params)
                if not rows:
                    empty_streak += 1
                    if empty_streak >= 2:      # 连续两次空(各自已重试 6 轮)= 真到底
                        break
                    offset += 200
                    continue
                empty_streak = 0
                for t in rows:
                    r = flatten(t)
                    if not r["track_id"] or r["track_id"] in seen:
                        continue
                    seen.add(r["track_id"])
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    added += 1; tag_new += 1
                f.flush()
                offset += 200
                if offset % 4000 == 0:
                    print(f"  [{label}] offset={offset} 本面新增 {tag_new} 总计 {len(seen)}",
                          flush=True)
                time.sleep(args.sleep)
            if tag_new:
                print(f"[{label}] 结束 offset={offset},新增 {tag_new},全局 {len(seen)}",
                      flush=True)
            if len(seen) >= args.max_tracks:
                print("[harvest] 达到 --max-tracks 上限,停止")
                break
    print(f"\n[harvest] 新增 {added},累计 {len(seen)} 条 → {meta_path}")


# -------------------------------------------------------------------- audit

def cmd_audit(args):
    out = Path(args.out)
    # 容错:harvest 还在追加时最后一行可能是半行;被 kill 过也会留残行。跳过即可。
    rows, bad = [], 0
    with open(out / META, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    print(f"[audit] 读入 {len(rows)} 条" + (f"(跳过 {bad} 条残行)" if bad else "") + "\n")

    # 艺术家级行为画像:批量灌库(大量曲目挤在极短窗口)是 AI 农场的典型形状
    by_artist = collections.defaultdict(list)
    for r in rows:
        by_artist[r["artist_id"]].append(r)
    burst = set()
    for aid, ts in by_artist.items():
        ds = sorted(d for d in (t["releasedate"] for t in ts) if len(d) == 10)
        if len(ts) >= args.burst_count and ds:
            span = (dt.date.fromisoformat(ds[-1]) - dt.date.fromisoformat(ds[0])).days
            if span <= args.burst_days:
                burst.add(aid)
    print(f"[audit] 疑似批量灌库账号: {len(burst)} 个 "
          f"(≥{args.burst_count} 首且跨度 ≤{args.burst_days} 天)")

    kept, stats = [], collections.Counter()
    for r in rows:
        reasons = []
        blob = " ".join([r["track_name"], r["artist_name"], r["mi_vartags"],
                         r["mi_genres"]])
        if AI_KEYWORDS.search(blob):
            reasons.append("关键词")
        if r["vocalinstrumental"] != "instrumental":
            reasons.append("非器乐")
        if not r["audiodownload"]:
            reasons.append("不可下载")
        if r["artist_id"] in burst:
            reasons.append("批量灌库账号")
        if r["releasedate"] >= AI_ERA_CUTOFF:
            reasons.append("AI后年代")
        for x in reasons:
            stats[x] += 1
        # 硬排除;"AI后年代" 单独不排除,只标风险(由 fetch 的 --max-risk 决定)
        hard = {"关键词", "非器乐", "不可下载", "批量灌库账号"}
        if hard & set(reasons):
            continue
        r["ai_risk"] = 1 if "AI后年代" in reasons else 0
        r["ai_reasons"] = "|".join(reasons)
        kept.append(r)

    print("\n[audit] 排除/标记计数:")
    for k, v in stats.most_common():
        print(f"   {k:14s} {v:7d}")

    yr = collections.Counter(r["year"] for r in kept)
    print(f"\n[audit] 通过 {len(kept)} 首 ({len({r['artist_id'] for r in kept})} 位艺术家)")
    print("   年份:", dict(sorted(yr.items())))
    print(f"   零风险(2023 前): {sum(1 for r in kept if r['ai_risk'] == 0)}")

    with open(out / CLEAN, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in FIELDS
                                          if c not in ("audio_id", "audio_path")])
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\n[audit] → {out / CLEAN}")


# ------------------------------------------------------------------ screen23

ARTISTS_API = "https://api.jamendo.com/v3.0/artists/"
POST23 = "clean_post2023.csv"


def _artist_page(ids, cid, tries=5):
    p = dict(client_id=cid, format="json", id="+".join(ids), limit=200)
    for a in range(tries):
        try:
            r = requests.get(ARTISTS_API, params=p, timeout=60)
            r.raise_for_status()
            res = r.json().get("results", [])
            if res or a == tries - 1:
                return res
        except Exception:
            if a == tries - 1:
                raise
        time.sleep(1.2 * (a + 1))
    return []


def cmd_screen23(args):
    """给 2023 年之后的曲目做二级筛查。

    2023 年后没有「结构上干净」这回事了(Suno 已上线),所以只能靠行为学。
    最硬的信号是**账号注册日期**:一个 2024 年注册、半年传了几百首的账号,
    和一个 2009 年注册、一直在传的账号,可信度差一个数量级。这就是 E13
    抓污染用的同一招,只是那次是手工查的。
    """
    cid = os.environ["JAMENDO_CLIENT_ID"]
    out = Path(args.out)

    rows, bad = [], 0
    with open(out / META, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1

    # 艺术家画像要用**全量**(含 2023 前)才算得准:有没有历史是关键信任信号
    by_artist = collections.defaultdict(list)
    for r in rows:
        by_artist[r["artist_id"]].append(r)

    cand = [r for r in rows
            if r["releasedate"] >= AI_ERA_CUTOFF
            and r["vocalinstrumental"] == "instrumental"
            and r["audiodownload"]
            and not AI_KEYWORDS.search(" ".join([r["track_name"], r["artist_name"],
                                                 r["mi_vartags"], r["mi_genres"]]))]
    aids = sorted({r["artist_id"] for r in cand if r["artist_id"]})
    print(f"[screen23] 2023+ 候选 {len(cand)} 首,涉及 {len(aids)} 位艺术家")

    join = {}
    for i in range(0, len(aids), 50):
        for a in _artist_page(aids[i:i + 50], cid):
            join[str(a["id"])] = a.get("joindate", "")
        if (i // 50) % 20 == 0:
            print(f"  查注册日期 {min(i + 50, len(aids))}/{len(aids)}", flush=True)
        time.sleep(0.2)
    print(f"[screen23] 拿到 {len(join)} 位艺术家的注册日期")

    kept, stats = [], collections.Counter()
    for r in cand:
        aid = r["artist_id"]
        mine = by_artist[aid]
        jd = join.get(aid, "")
        has_history = any(t["releasedate"] < AI_ERA_CUTOFF for t in mine)
        ds = sorted(d for d in (t["releasedate"] for t in mine) if len(d) == 10)
        span = ((dt.date.fromisoformat(ds[-1]) - dt.date.fromisoformat(ds[0])).days
                if len(ds) >= 2 else 0)

        reasons = []
        if jd and jd >= AI_ERA_CUTOFF:
            reasons.append("账号2023后注册")
        if not jd:
            reasons.append("查不到注册日期")
        if len(mine) >= args.burst_count23 and span <= args.burst_days23:
            reasons.append("短期批量上传")
        # 上传速率:注册以来平均每天几首。真人极少长期 >0.5
        if jd and len(jd) == 10:
            days = max((dt.date.today() - dt.date.fromisoformat(jd)).days, 1)
            if len(mine) / days > args.max_velocity:
                reasons.append("上传速率异常")
        if not has_history:
            reasons.append("无2023前作品")

        for x in reasons:
            stats[x] += 1
        hard = {"账号2023后注册", "短期批量上传", "上传速率异常", "查不到注册日期"}
        if hard & set(reasons):
            stats["__踢掉"] += 1
            continue
        # 留下来的都是老账号的新作品。有历史的记 1,没历史的记 2(更弱)
        r["ai_risk"] = 1 if has_history else 2
        r["ai_reasons"] = "|".join(reasons) or "老账号新作"
        r["artist_joindate"] = jd
        kept.append(r)

    print("\n[screen23] 命中计数:")
    for k, v in stats.most_common():
        print(f"   {k:16s} {v:6d}")
    yr = collections.Counter(r["year"] for r in kept)
    print(f"\n[screen23] 通过 {len(kept)} 首 ({len({r['artist_id'] for r in kept})} 位艺术家)")
    print("   年份:", dict(sorted(yr.items())))
    print(f"   risk1(老账号+有历史) {sum(1 for r in kept if r['ai_risk'] == 1)}"
          f"   risk2(老账号无历史) {sum(1 for r in kept if r['ai_risk'] == 2)}")

    cols = [c for c in FIELDS if c not in ("audio_id", "audio_path")] + ["artist_joindate"]
    with open(out / POST23, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n[screen23] → {out / POST23}")


# -------------------------------------------------------------------- fetch

def _dl(r, dest: Path, tries=3):
    """下一首,带重试。

    网络抖动(ProxyError/ConnectionError/ReadTimeout/SSLError)不等于「这首不可用」,
    一次失败就丢弃会白白损失约 1.5% 的曲目;而 404 之类的永久错误重试也没用,
    所以只对网络类异常退避重试。
    """
    if _is_valid_mp3(dest):
        return True, ""
    last = ""
    for a in range(tries):
        try:
            resp = requests.get(r["audiodownload"], timeout=180)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            break
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", 0)
            if code and code < 500:                 # 4xx 永久错,重试无意义
                return False, f"下载失败 HTTP{code}"
            last = f"下载失败 HTTP{code}"
        except Exception as e:
            last = f"下载失败 {type(e).__name__}"
        if a == tries - 1:
            return False, last
        time.sleep(2 ** a)
    if not _is_valid_mp3(dest):
        dest.unlink(missing_ok=True)
        return False, "非法mp3"
    return True, ""


def cmd_fetch(args):
    out = Path(args.out)
    src = Path(args.clean_file) if args.clean_file else out / CLEAN
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    print(f"[fetch] 来源 {src.name}")
    rows = [r for r in rows if int(r["ai_risk"]) <= args.max_risk]

    # 每位艺术家上限,保人群多样性。
    # 注意:不能按 artist_id 排序再截断——artist_id 就是注册先后,那样等于
    # 系统性偏向老账号。改成固定种子洗牌(可复现),风险低的仍排前面。
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    rows.sort(key=lambda r: int(r["ai_risk"]))     # 稳定排序,组内保持洗牌顺序
    per, picked = collections.Counter(), []
    for r in rows:
        if per[r["artist_id"]] >= args.per_artist_cap:
            continue
        per[r["artist_id"]] += 1
        picked.append(r)
        if args.target and len(picked) >= args.target:
            break
    print(f"[fetch] 选出 {len(picked)} 首 ({len(per)} 位艺术家,每人≤{args.per_artist_cap})")

    audio = out / args.subdir; audio.mkdir(parents=True, exist_ok=True)
    mpath = out / (MANIFEST if args.subdir == "audio"
                   else f"manifest_{args.subdir}.csv")
    done = set()
    if mpath.exists():
        for r in csv.DictReader(open(mpath, encoding="utf-8")):
            done.add(r["track_id"])
        print(f"[续跑] manifest 已有 {len(done)} 首")

    jobs = [(r, audio / f"jam_{r['track_id']}.mp3")
            for r in picked if r["track_id"] not in done]
    stats = collections.Counter()
    new = not mpath.exists()
    with open(mpath, "a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_dl, r, d): (r, d) for r, d in jobs}
            for n, fut in enumerate(cf.as_completed(futs), 1):
                r, d = futs[fut]
                ok, err = fut.result()
                if not ok:
                    stats[err] += 1
                    continue
                row = {k: r.get(k, "") for k in FIELDS}
                row["audio_id"] = f"jam_{r['track_id']}"
                row["audio_path"] = str(d)
                w.writerow(row); mf.flush()
                stats["入库"] += 1
                if n % 200 == 0:
                    print(f"  {n}/{len(jobs)}  {dict(stats)}", flush=True)
    print(f"\n[fetch] {dict(stats)} → {mpath}")


def main():
    ap = argparse.ArgumentParser(description="Jamendo 器乐扫库:不是 AI + 是器乐,只要这两条")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest"); h.add_argument("--out", required=True)
    h.add_argument("--max-tracks", type=int, default=100000)
    h.add_argument("--tags", nargs="+", default=SWEEP_TAGS)
    h.add_argument("--years", nargs="*", type=int,
                   default=list(range(2015, 2027)),
                   help="额外按年份切片重扫的年份(不切时老歌会霸榜);传空列表关闭")
    h.add_argument("--sleep", type=float, default=0.25)

    a = sub.add_parser("audit"); a.add_argument("--out", required=True)
    a.add_argument("--burst-count", type=int, default=25,
                   help="同一账号曲目数达到此值且日期跨度极短 → 判为批量灌库")
    a.add_argument("--burst-days", type=int, default=60)

    s23 = sub.add_parser("screen23"); s23.add_argument("--out", required=True)
    s23.add_argument("--burst-count23", type=int, default=15)
    s23.add_argument("--burst-days23", type=int, default=45)
    s23.add_argument("--max-velocity", type=float, default=0.5,
                     help="注册以来平均每天上传首数上限;真人极少长期超过 0.5")

    f = sub.add_parser("fetch"); f.add_argument("--out", required=True)
    f.add_argument("--target", type=int, default=0, help="0 = 全下")
    f.add_argument("--per-artist-cap", type=int, default=5)
    f.add_argument("--max-risk", type=int, default=0,
                   help="0 = 只要 2023 年前(结构上无 AI);1 = 也要 2023 后的")
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--seed", type=int, default=0, help="选曲洗牌种子,固定可复现")
    f.add_argument("--clean-file", default="", help="改从别的审计结果读(如 clean_post2023.csv)")
    f.add_argument("--subdir", default="audio", help="音频子目录,分层存放用")

    args = ap.parse_args()
    if not os.environ.get("JAMENDO_CLIENT_ID"):
        sys.exit("先 export JAMENDO_CLIENT_ID=…(别写进代码)")
    {"harvest": cmd_harvest, "audit": cmd_audit, "screen23": cmd_screen23,
     "fetch": cmd_fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
