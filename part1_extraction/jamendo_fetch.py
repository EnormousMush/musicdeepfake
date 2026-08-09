"""
Jamendo 现代人类对照语料采集(年代混淆拆旗行动,2026-08-09 立项)。

动机:H1 消融显示手工特征 8.0% EER 几乎全由音色/频谱族扛(独自 9.6%),
而音色族最大嫌疑 = 2010s FMA vs 2025 Suno 的年代/制作差。本脚本从 Jamendo
(CC 授权曲库,带上传日期)采集 2024–2026 年发行的人类音乐,与 FMA 同为
independent/CC 生态——**唯一变化的是年代**,是干净的对照。

流程:API 按 genre 拉候选池(releasedate 过滤)→ 本地抽样(每 artist 上限 2 首,
保群体多样性;每 genre 目标 375 首,8 genre 共 3000,对齐 suno 批)→ 下载 mp3
→ 产出 manifest CSV。后续走既有管线:crossgen_prep 共同规格(10s/16k/mono/LUFS-23)。

前置:在 https://devportal.jamendo.com 免费注册拿 client_id(不要提交进 git):
  export JAMENDO_CLIENT_ID=xxxxxxxx

Usage(Mac 本地即可,烟测先行):
  python part1_extraction/jamendo_fetch.py --out "/Volumes/Seagate /honors_paper/1_corpora_real/jamendo2025" --limit 5
  python part1_extraction/jamendo_fetch.py --out "/Volumes/Seagate /honors_paper/1_corpora_real/jamendo2025"

断点续跑:已存在且校验通过的 mp3 跳过;manifest 每 genre 增量落盘。
授权:逐曲记录 license_ccurl;仅本地研究分析,不再分发,ND/NC 均可用。
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

API = "https://api.jamendo.com/v3.0/tracks/"
GENRES = ["blues", "classical", "country", "electronic", "hiphop", "jazz", "pop", "rock"]
# Jamendo fuzzytags 词表与我们的 genre 名对齐(hiphop 在 Jamendo 常写 hiphop/rap)
FUZZY = {g: g for g in GENRES}
FUZZY["hiphop"] = "hiphop rap"    # 多值用空格分隔;"+"会被编码成 %2B 变非法参数

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")


def _is_valid_mp3(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10_000:  # <10KB 视为残片
        return False
    with open(path, "rb") as f:
        header = f.read(3)
    return any(header[: len(m)] == m for m in _MP3_MAGIC)


def _get_page(params, tries=4):
    """一页,带重试:Jamendo 会间歇性返回空结果(无错误码),空页也重试确认。"""
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
        time.sleep(2.0 * (attempt + 1))
    return []


def api_pool(client_id, genre, date_lo, date_hi, pool_target, sleep_s=0.8):
    """按 genre 拉候选池:releasedate 过滤 + 时长门槛 + 可下载。返回 track dict 列表。"""
    out, offset = [], 0
    while len(out) < pool_target:
        rows = _get_page(dict(
            client_id=client_id, format="json", limit=200, offset=offset,
            fuzzytags=FUZZY[genre], datebetween=f"{date_lo}_{date_hi}",
            durationbetween="60_600", audioformat="mp32",
            include="musicinfo licenses", order="popularity_total",
        ))
        if not rows:
            break
        for t in rows:
            if t.get("audiodownload_allowed") and t.get("audiodownload"):
                out.append(t)
        offset += 200
        time.sleep(sleep_s)
    return out


def sample_diverse(pool, n_target, per_artist_cap, seed=0):
    """人群多样性优先:每 artist 上限 cap 首;池内随机(seed 可复现)。
    池子小凑不满时,逐步放宽上限兜底(cap→2cap→4cap),放宽情况打印告警。"""
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
            picked.append(t); chosen.add(i)
            if len(picked) >= n_target:
                return picked
        if cap > per_artist_cap:
            print(f"  警告:歌手上限放宽到 {cap} 仍只凑到 {len(picked)}/{n_target}")
    return picked


FIELDS = ["audio_id", "genre", "track_id", "track_name", "artist_id", "artist_name",
          "releasedate", "duration", "license_ccurl", "audiodownload", "audio_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-genre", type=int, default=375)
    ap.add_argument("--per-artist-cap", type=int, default=2)
    ap.add_argument("--date-lo", default="2024-01-01")
    ap.add_argument("--date-hi", default="2026-08-01")
    ap.add_argument("--pool-mult", type=int, default=4, help="候选池 = per_genre × 此倍数")
    ap.add_argument("--limit", type=int, default=0, help="烟测:每 genre 只下前 N 首")
    args = ap.parse_args()

    client_id = os.environ.get("JAMENDO_CLIENT_ID", "")
    if not client_id:
        sys.exit("先 export JAMENDO_CLIENT_ID=…(devportal.jamendo.com 免费注册)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest_jamendo.csv"
    done_ids = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for r in csv.DictReader(f):
                if _is_valid_mp3(Path(r["audio_path"])):
                    done_ids.add(r["audio_id"])
        print(f"resume: {len(done_ids)} already valid")

    new_manifest = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=FIELDS)
        if new_manifest:
            w.writeheader()
        for genre in GENRES:
            n_target = args.limit or args.per_genre
            print(f"[{genre}] pulling pool …", flush=True)
            pool = api_pool(client_id, genre, args.date_lo, args.date_hi,
                            pool_target=args.per_genre * args.pool_mult)
            picked = sample_diverse(pool, n_target, args.per_artist_cap)
            n_artists = len({t["artist_id"] for t in picked})
            print(f"[{genre}] pool {len(pool)} -> picked {len(picked)} "
                  f"({n_artists} artists)", flush=True)
            gdir = out_dir / genre
            gdir.mkdir(exist_ok=True)
            for j, t in enumerate(picked):
                audio_id = f"jam_{genre}_{j:04d}"
                if audio_id in done_ids:
                    continue
                dest = gdir / f"{audio_id}.mp3"
                if not _is_valid_mp3(dest):
                    try:
                        r = requests.get(t["audiodownload"], timeout=180)
                        r.raise_for_status()
                        dest.write_bytes(r.content)
                    except Exception as exc:
                        print(f"  {audio_id}: download error {exc}", flush=True)
                        continue
                    if not _is_valid_mp3(dest):
                        print(f"  {audio_id}: invalid mp3, skipped", flush=True)
                        dest.unlink(missing_ok=True)
                        continue
                    time.sleep(0.3)
                w.writerow(dict(
                    audio_id=audio_id, genre=genre, track_id=t.get("id", ""),
                    track_name=t.get("name", ""), artist_id=t.get("artist_id", ""),
                    artist_name=t.get("artist_name", ""),
                    releasedate=t.get("releasedate", ""), duration=t.get("duration", ""),
                    license_ccurl=t.get("license_ccurl", ""),
                    audiodownload=t.get("audiodownload", ""), audio_path=str(dest)))
                mf.flush()
                if (j + 1) % 25 == 0:
                    print(f"  [{genre}] {j+1}/{len(picked)}", flush=True)
    print(f"done -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
