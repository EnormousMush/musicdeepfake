"""
Jamendo 年代分层器乐语料采集(人类数据集扩张,2026-09-18 立项)。

动机:现有人类语料的年代分布是断的——
    FMA 2008–2017 · MTG-Jamendo 2005–2017 · MagnaTagATune 2000s
    ……(2018–2023 空白)……
    jamendo2025 2024–2026
中间这 6 年正好是「现代制作水准已经到位、但生成式音乐还没出现」的窗口。
它是分离两个混淆因子的关键对照:**年代/制作差** vs **AI 签名**。
E13 教训说明污染会直接翻转结论(移除 32 首把 suno TPR@5% 从 47.3% 抬到 84.9%),
而 2018–2022 层在结构上就是干净的——Suno 2023 年底才上线。

与 jamendo_fetch.py 的区别(那个脚本拉的是 2024–26 层,已完成 3000 首):
  1. **按 (genre × year) 分层**,每个年份格子独立配额,年代分布可控而非碰运气
  2. **硬过滤 musicinfo.vocalinstrumental == "instrumental"**,入库即 100% 器乐,
     不再像 jamendo2025 那样事后补审计(那批 3000 首里只有 2050 首是器乐)
  3. **全局去重**:跳过 jamendo2025 manifest 里已有的 track_id
  4. manifest 记录完整 musicinfo(器乐标记/原声电声/速度/genre/乐器/vartags)+ 授权

前置(client_id 在 https://devportal.jamendo.com 免费注册,**不要提交进 git**):
  export JAMENDO_CLIENT_ID=xxxxxxxx

Usage:
  # 烟测:每个 (genre, year) 只拉 2 首,验证通道和落盘
  python part1_extraction/jamendo_eras.py \
      --out "/Volumes/Seagate /honors_paper/1_corpora_real/jamendo_eras" --limit 2

  # 正式:2018–2022,8 genre × 5 year × 50 = 2000 首器乐
  python part1_extraction/jamendo_eras.py \
      --out "/Volumes/Seagate /honors_paper/1_corpora_real/jamendo_eras" \
      --per-cell 50

断点续跑:已存在且校验通过的 mp3 跳过;manifest 每首增量落盘,随时可 Ctrl-C。
授权:逐曲记录 license_ccurl;仅本地研究分析,不再分发,ND/NC 均可用。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import collections
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

API = "https://api.jamendo.com/v3.0/tracks/"
GENRES = ["blues", "classical", "country", "electronic", "hiphop", "jazz", "pop", "rock"]
FUZZY = {g: g for g in GENRES}
FUZZY["hiphop"] = "hiphop rap"     # 多值用空格分隔;"+" 会被编码成 %2B 变非法参数

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")

FIELDS = [
    "audio_id", "year", "genre", "track_id", "track_name",
    "artist_id", "artist_name", "album_id", "releasedate", "duration",
    "vocalinstrumental", "acousticelectric", "speed",
    "mi_genres", "mi_instruments", "mi_vartags",
    "license_ccurl", "audiodownload", "audio_path",
]


def _is_valid_mp3(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:   # <10KB 视为残片
        return False
    with open(path, "rb") as f:
        return any(f.read(3)[: len(m)] == m for m in _MP3_MAGIC)


def _get_page(params, tries=5):
    """一页,带重试。

    Jamendo 会**间歇性返回空结果且不报错**(headers.status 仍是 success),
    所以空页也必须重试确认,不能当成「翻到底了」。这是实测踩过的坑。
    """
    for attempt in range(tries):
        try:
            r = requests.get(API, params=params, timeout=60)
            r.raise_for_status()
            body = r.json()
            if body.get("headers", {}).get("status") != "success":
                raise RuntimeError(f"API error: {body.get('headers')}")
            rows = body.get("results", [])
            if rows or attempt == tries - 1:
                return rows
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    return []


def api_pool(client_id, genre, year, pool_target, sleep_s=0.6):
    """拉某个 (genre, year) 的器乐候选池。

    fuzzytags 同时给 genre 和 "instrumental":实测 instrumental 标签的器乐命中率
    100%,genre 标签 92–98%。即便如此仍然逐曲用 musicinfo 复核,标签只是提效。
    """
    out, offset, seen = [], 0, set()
    tags = f"{FUZZY[genre]} instrumental"
    while len(out) < pool_target:
        rows = _get_page(dict(
            client_id=client_id, format="json", limit=200, offset=offset,
            fuzzytags=tags, datebetween=f"{year}-01-01_{year}-12-31",
            durationbetween="60_600", audioformat="mp32",
            include="musicinfo licenses", order="popularity_total",
        ))
        if not rows:
            break
        for t in rows:
            tid = str(t.get("id", ""))
            if tid in seen:
                continue
            seen.add(tid)
            # 硬门槛:必须是器乐 + 可下载
            if t.get("musicinfo", {}).get("vocalinstrumental") != "instrumental":
                continue
            if not (t.get("audiodownload_allowed") and t.get("audiodownload")):
                continue
            out.append(t)
        offset += 200
        time.sleep(sleep_s)
    return out


def _download_one(audio_id, t, dest: Path):
    """下一首。返回 (ok, err)。已存在且校验通过则直接算成功(断点续跑)。"""
    if _is_valid_mp3(dest):
        return True, ""
    try:
        r = requests.get(t["audiodownload"], timeout=180)
        r.raise_for_status()
        dest.write_bytes(r.content)
    except Exception as exc:
        return False, f"下载失败 {exc}"
    if not _is_valid_mp3(dest):
        dest.unlink(missing_ok=True)
        return False, "非法mp3 丢弃"
    return True, ""


def sample_diverse(pool, n_target, per_artist_cap, seed=0):
    """人群多样性优先:每 artist 上限 cap 首;池内随机(seed 可复现)。
    池子小凑不满时逐步放宽上限兜底(cap→2cap→4cap),放宽情况打印告警。"""
    rng = np.random.default_rng(seed)
    order = [int(i) for i in rng.permutation(len(pool))]
    picked, per_artist, chosen = [], {}, set()
    for cap in (per_artist_cap, per_artist_cap * 2, per_artist_cap * 4):
        for i in order:
            if i in chosen:
                continue
            t = pool[i]
            a = str(t.get("artist_id", ""))
            if per_artist.get(a, 0) >= cap:
                continue
            per_artist[a] = per_artist.get(a, 0) + 1
            picked.append(t)
            chosen.add(i)
            if len(picked) >= n_target:
                return picked
        if cap > per_artist_cap and len(picked) < n_target:
            print(f"    歌手上限放宽到 {cap},仍只凑到 {len(picked)}/{n_target}")
    return picked


def load_existing_track_ids(paths) -> set:
    """读已有语料的 manifest,拿到要排除的 track_id(跨语料去重)。"""
    ids = set()
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                tid = row.get("track_id", "")
                if tid:
                    ids.add(str(tid))
        print(f"[去重] {p.name}: 累计排除 {len(ids)} 个 track_id")
    return ids


def row_from(t, audio_id, year, genre, dest) -> dict:
    mi = t.get("musicinfo", {}) or {}
    tags = mi.get("tags", {}) or {}
    return dict(
        audio_id=audio_id, year=year, genre=genre,
        track_id=t.get("id", ""), track_name=t.get("name", ""),
        artist_id=t.get("artist_id", ""), artist_name=t.get("artist_name", ""),
        album_id=t.get("album_id", ""), releasedate=t.get("releasedate", ""),
        duration=t.get("duration", ""),
        vocalinstrumental=mi.get("vocalinstrumental", ""),
        acousticelectric=mi.get("acousticelectric", ""),
        speed=mi.get("speed", ""),
        mi_genres="|".join(tags.get("genres", []) or []),
        mi_instruments="|".join(tags.get("instruments", []) or []),
        mi_vartags="|".join(tags.get("vartags", []) or []),
        license_ccurl=t.get("license_ccurl", ""),
        audiodownload=t.get("audiodownload", ""), audio_path=str(dest),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Jamendo 年代分层器乐采集(每个 genre×year 独立配额,入库即 100% 器乐)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--year-lo", type=int, default=2018)
    ap.add_argument("--year-hi", type=int, default=2022)
    ap.add_argument("--per-cell", type=int, default=50, help="每个 (genre, year) 目标首数")
    ap.add_argument("--per-artist-cap", type=int, default=2)
    ap.add_argument("--pool-mult", type=int, default=4, help="候选池 = per_cell × 此倍数")
    ap.add_argument("--genres", nargs="+", default=GENRES)
    ap.add_argument("--limit", type=int, default=0, help="烟测:每格只下前 N 首")
    ap.add_argument("--workers", type=int, default=6,
                    help="并行下载连接数(Jamendo CDN 单连接限速;别调太高)")
    ap.add_argument("--exclude-manifest", nargs="*", default=[],
                    help="要排除 track_id 的已有 manifest(默认自动带上 jamendo2025)")
    args = ap.parse_args()

    client_id = os.environ.get("JAMENDO_CLIENT_ID", "")
    if not client_id:
        sys.exit("先 export JAMENDO_CLIENT_ID=…(devportal.jamendo.com 免费注册,别写进代码)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_jamendo_eras.csv"

    # 跨语料去重:默认把同一块盘上的 jamendo2025 带上
    excl = list(args.exclude_manifest)
    sibling = out_dir.parent / "jamendo2025" / "manifest_jamendo.csv"
    if not excl and sibling.exists():
        excl = [str(sibling)]
    exclude_ids = load_existing_track_ids(excl)

    # 断点续跑
    done_ids, have_tracks = set(), set()
    if manifest_path.exists():
        with open(manifest_path, newline="") as f:
            for r in csv.DictReader(f):
                have_tracks.add(str(r["track_id"]))
                if _is_valid_mp3(Path(r["audio_path"])):
                    done_ids.add(r["audio_id"])
        print(f"[续跑] 已有 {len(done_ids)} 首有效")
    exclude_ids |= have_tracks

    years = list(range(args.year_lo, args.year_hi + 1))
    n_target = args.limit or args.per_cell
    print(f"[计划] {len(args.genres)} genre × {len(years)} year × {n_target} 首 "
          f"= {len(args.genres) * len(years) * n_target} 首上限\n")

    stats = collections.Counter()
    new_manifest = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=FIELDS)
        if new_manifest:
            w.writeheader()

        for year in years:
            ydir = out_dir / str(year)
            ydir.mkdir(exist_ok=True)
            for genre in args.genres:
                print(f"[{year}/{genre}] 拉池子 …", flush=True)
                pool = api_pool(client_id, genre, year,
                                pool_target=args.per_cell * args.pool_mult)
                pool = [t for t in pool if str(t.get("id", "")) not in exclude_ids]
                picked = sample_diverse(pool, n_target, args.per_artist_cap)
                n_art = len({t["artist_id"] for t in picked})
                print(f"[{year}/{genre}] 池 {len(pool)} → 选 {len(picked)} ({n_art} 位艺术家)",
                      flush=True)

                # 下载并行(Jamendo CDN 单连接限速明显),manifest 仍由主线程串行写,
                # 保证 CSV 不被并发写坏、且中断时已写的行一定对应校验通过的文件。
                jobs = []
                for j, t in enumerate(picked):
                    audio_id = f"jam{year}_{genre}_{j:04d}"
                    if audio_id in done_ids:
                        continue
                    jobs.append((audio_id, t, ydir / f"{audio_id}.mp3"))

                with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {ex.submit(_download_one, aid, t, dest): (aid, t, dest)
                            for aid, t, dest in jobs}
                    for n, fut in enumerate(cf.as_completed(futs), 1):
                        aid, t, dest = futs[fut]
                        ok, err = fut.result()
                        if not ok:
                            print(f"    {aid}: {err}", flush=True)
                            stats[err.split()[0]] += 1
                            continue
                        w.writerow(row_from(t, aid, year, genre, dest))
                        mf.flush()
                        exclude_ids.add(str(t.get("id", "")))
                        stats["入库"] += 1
                        if n % 25 == 0:
                            print(f"    [{year}/{genre}] {n}/{len(jobs)}", flush=True)

    print(f"\n[完成] {dict(stats)} → {manifest_path}")


if __name__ == "__main__":
    main()
