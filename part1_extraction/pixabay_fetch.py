"""
Pixabay 器乐人类语料采集(人类数据集扩张,2026-09-18 立项)。

动机:P2 需要 ≥12 个互相独立的人类数据集,目前只用了 4–5 个。Pixabay Music
是候选之一——量大、CC0/Pixabay License、有 tag 体系。但它同时是**污染最重**的
候选:2023 年之后站内涌入大量 AI 生成音乐,且平台不保证逐曲标注。因此本脚本
把「逐曲记录 AI 嫌疑信号」当成一等公民,而不是事后补救(参见 E13 教训:
jamendo 12 个账号 69 首污染,移除 32 首把 suno TPR@5% 从 47.3% 抬到 84.9%)。

访问现状(2026-09-18 实测,见 README 段落):
  * pixabay.com/music/* 网页 → Cloudflare 质询,对任何脚本返回 403
    (响应头 cf-mitigated: challenge)
  * robots.txt 公告的 sitemap.xml.gz → 同样被质询
  * 官方 API(pixabay.com/api/)→ 不质询,但**只有 images 和 videos 两个
    端点,没有 music**
  故:目前不存在「被站方认可的机器可读音乐通道」。本脚本按正规爬虫写死——
  具名 UA、遵守 robots、限速、断点续跑——遇到质询立即**停机并报告**,
  不做任何指纹伪装。要让它跑起来,需要站方做下面任一件事(见 --help 尾部):
    1. Cloudflare WAF 加一条 allow 规则(按 UA 或按我们的出口 IP)——最省事
    2. 发一个 API key / bearer token,我们用 --api-key / --bearer 带上
    3. 直接给批量导出(最优:省掉全部爬取,且能附带逐曲 AI 标记)

前置:
  export PIXABAY_CONTACT="durunbao0620@gmail.com"      # 必填,写进 UA
  export PIXABAY_API_KEY=xxxx                          # 站方给了才需要
  export PIXABAY_EXTRA_HEADER="X-Pixabay-Allow: xxxx"  # 站方给了才需要

Usage:
  # 第一步:只探通道,不抓数据(先跑这个)
  python part1_extraction/pixabay_fetch.py preflight

  # 第二步:枚举候选曲目元数据(不下音频)
  python part1_extraction/pixabay_fetch.py discover \
      --queries ambient piano cinematic --pages 5 \
      --out "/Volumes/Seagate /honors_paper/1_corpora_real/pixabay"

  # 第三步:下载音频
  python part1_extraction/pixabay_fetch.py fetch \
      --out "/Volumes/Seagate /honors_paper/1_corpora_real/pixabay" --limit 50

断点续跑:discover 写 JSONL(按 track_id 去重),fetch 跳过已存在且校验通过的 mp3。
授权:逐曲记录 license / 上传者 / 上传日期 / 全部 tag / AI 嫌疑信号;仅本地研究分析,不再分发。
"""
from __future__ import annotations   # 本机是 Python 3.9,需要它才能写 X | None

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.robotparser
from pathlib import Path

import requests

BASE = "https://pixabay.com"
ROBOTS = f"{BASE}/robots.txt"
MUSIC_SEARCH = f"{BASE}/music/search/{{query}}/"

# robots.txt 明令禁止的过滤参数——本脚本一律不带,靠 query 词本身分流
FORBIDDEN_PARAMS = {
    "orientation", "manual_search", "min_width", "min_height", "date", "colors",
    "order", "animation", "resolution_hd", "genre", "mood", "movement", "theme",
    "cat", "layout", "per_page", "content_type", "vertex_count",
    "with_animations", "duration", "spell_check",
}

# AI 嫌疑信号:标题/标签/描述里出现即记一分。不做硬删除,只打分留给下游决策。
AI_MARKERS = re.compile(
    r"\b(ai[\s\-_]?generated|generated[\s\-_]?by[\s\-_]?ai|suno|udio|mubert|soundraw|"
    r"aiva|boomy|stable[\s\-_]?audio|musicgen|riffusion|text[\s\-_]?to[\s\-_]?music|"
    r"ai[\s\-_]?music|neural[\s\-_]?composed)\b",
    re.I,
)
# 器乐正信号 / 人声负信号(Pixabay 无 vocal/instrumental 字段,只能靠 tag 近似)
INSTRUMENTAL_TAGS = {"instrumental", "no vocals", "background music", "backing track"}
VOCAL_TAGS = {"vocal", "vocals", "singing", "singer", "choir", "acapella",
              "a cappella", "lyrics", "rap", "voice"}

_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa")


class AccessBlocked(RuntimeError):
    """站方的机器人质询。不重试、不伪装——直接停机报告。"""


class RobotsDenied(RuntimeError):
    pass


# ---------------------------------------------------------------- 传输层

def build_session(contact: str, extra_header: str | None) -> requests.Session:
    """具名会话。UA 里写明用途和联系方式,便于站方在日志里认出我们并放行。"""
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            f"UChicagoMusicResearchBot/0.1 "
            f"(academic AI-music-detection research; +mailto:{contact})"
        ),
        "From": contact,
        "X-Research-Purpose": "non-commercial academic dataset for AI-music detection",
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en",
    })
    if extra_header:
        name, _, value = extra_header.partition(":")
        if value:
            s.headers[name.strip()] = value.strip()
    return s


def is_challenge(resp: requests.Response) -> bool:
    if "cf-mitigated" in {k.lower() for k in resp.headers}:
        return True
    if resp.status_code in (403, 503):
        body = resp.text[:4000].lower()
        return ("challenges.cloudflare.com" in body or "__cf_chl" in body
                or "just a moment" in body)
    return False


class Fetcher:
    """限速 + 退避 + robots 强制 + 质询停机。单连接,不并发。"""

    def __init__(self, session, delay=2.0, jitter=1.0, verbose=True):
        self.s = session
        self.delay = delay
        self.jitter = jitter
        self.verbose = verbose
        self._last = 0.0
        self._rp = None

    def _sleep(self):
        wait = self.delay + random.uniform(0, self.jitter) - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def load_robots(self):
        self._sleep()
        r = self.s.get(ROBOTS, timeout=30)
        if is_challenge(r):
            raise AccessBlocked("robots.txt 本身就被质询,通道完全关闭")
        r.raise_for_status()
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(r.text.splitlines())
        self._rp = rp
        if self.verbose:
            print(f"[robots] 已加载 {len(r.text.splitlines())} 行规则")
        return rp

    def allowed(self, url: str) -> bool:
        if self._rp is None:
            self.load_robots()
        return self._rp.can_fetch(self.s.headers["User-Agent"], url)

    def get(self, url: str, params=None, tries=3):
        if params:
            bad = FORBIDDEN_PARAMS & set(params)
            if bad:
                raise RobotsDenied(f"robots.txt 禁止的查询参数: {sorted(bad)}")
        if not self.allowed(url):
            raise RobotsDenied(f"robots.txt 不允许抓取: {url}")

        for attempt in range(tries):
            self._sleep()
            r = self.s.get(url, params=params, timeout=60, allow_redirects=True)
            if is_challenge(r):
                raise AccessBlocked(
                    f"{url} 返回机器人质询(HTTP {r.status_code}, "
                    f"cf-mitigated={r.headers.get('cf-mitigated', '-')})"
                )
            if r.status_code == 429 or 500 <= r.status_code < 600:
                back = (2 ** attempt) * 5
                if self.verbose:
                    print(f"[retry] HTTP {r.status_code},{back}s 后重试 ({attempt + 1}/{tries})")
                time.sleep(back)
                continue
            r.raise_for_status()
            return r
        raise RuntimeError(f"{url} 连续 {tries} 次失败")


# ---------------------------------------------------------------- 解析层

def extract_payload(html: str) -> list[dict]:
    """
    从搜索页 HTML 里抠出曲目 JSON。

    Pixabay 前端是 SSR+hydration,数据藏在内联 script 里。我们无法在拿到通道前
    验证确切的键名,所以这里按「多策略 + 结构化兜底」写:任一策略命中即用,
    全不命中就退回正则扫 /music/<slug>-<id>/ 链接,并把原始 HTML 落盘供人工对照。
    """
    cands: list[dict] = []

    # 策略 1:<script type="application/json"> …(Nuxt/Next 的标准落点)
    for m in re.finditer(
        r'<script[^>]+type="application/json"[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            cands.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass

    # 策略 2:window.__NUXT__ / __INITIAL_STATE__ = {...}
    for m in re.finditer(
        r"window\.(?:__NUXT__|__INITIAL_STATE__)\s*=\s*(\{.*?\});?\s*</script>", html, re.S
    ):
        try:
            cands.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass

    tracks: list[dict] = []
    seen: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            # 一个 dict 同时有 id 和 (audio|audio_url|sources) 就当作曲目
            keys = {k.lower() for k in node}
            if "id" in keys and keys & {"audio", "audio_url", "sources", "mp3", "src"}:
                tid = str(node.get("id"))
                if tid not in seen:
                    seen.add(tid)
                    tracks.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for c in cands:
        walk(c)

    # 策略 3(兜底):正则扫曲目页链接,只拿到 id+slug,详情留给 fetch 阶段
    if not tracks:
        for m in re.finditer(r'href="(/music/[^"]*?-(\d+)/)"', html):
            tid = m.group(2)
            if tid not in seen:
                seen.add(tid)
                tracks.append({"id": tid, "pageURL": BASE + m.group(1), "_partial": True})

    return tracks


def normalize(raw: dict) -> dict:
    """把站方原始字段压成我们自己的行格式;缺字段留空,不猜。"""
    def first(*names, default=""):
        for n in names:
            for k, v in raw.items():
                if k.lower() == n and v not in (None, ""):
                    return v
        return default

    tags = first("tags", "keywords", default=[])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [str(t).lower() for t in tags]

    title = str(first("title", "name"))
    desc = str(first("description", "summary"))
    blob = " ".join([title, desc, " ".join(tags)])

    tagset = set(tags)
    if tagset & VOCAL_TAGS:
        instr = "vocal"
    elif tagset & INSTRUMENTAL_TAGS:
        instr = "instrumental"
    else:
        instr = "unknown"          # Pixabay 无该字段,绝大多数会落在这里

    return {
        "track_id": str(first("id")),
        "page_url": str(first("pageurl", "page_url", "url")),
        "audio_url": str(first("audio_url", "audio", "mp3", "src")),
        "title": title,
        "uploader": str(first("user", "username", "author", "artist")),
        "uploaded": str(first("uploaded", "created", "date", "releasedate")),
        "duration_s": first("duration", default=""),
        "license": str(first("license", default="Pixabay Content License")),
        "tags": "|".join(tags),
        "instrumental_guess": instr,
        "ai_suspect": int(bool(AI_MARKERS.search(blob))),
        "ai_markers": "|".join(sorted({m.group(0).lower() for m in AI_MARKERS.finditer(blob)})),
    }


# ---------------------------------------------------------------- 子命令

BLOCKED_ADVICE = """
────────────────────────────────────────────────────────────────────
通道被站方的机器人质询挡住了。这不是限速问题,重试或换时间段都没用——
Cloudflare 质询由**站方**配置,只能站方解除。

本脚本刻意不做指纹伪装(不改 navigator.webdriver、不 exclude enable-automation、
不禁 AutomationControlled)。那类改动的唯一作用就是让站方的机器人识别失效,
既违反 Pixabay 服务条款,也会让「我们拿到了授权」这件事无从体现在流量里
——站方日志里看到的仍然是一个伪装成人的匿名浏览器。

既然已经跟对方开过会,请把下面任一条落实(按省事程度排序):

  1. Cloudflare WAF allow 规则(最省事,对方 2 分钟能做完)
     给他们:UA 字符串 "UChicagoMusicResearchBot/0.1"
             或我们的出口 IP(curl ifconfig.me 拿到后给他们)
     做完这个,本脚本原样就能跑。

  2. API key 或 bearer token
     拿到后:--api-key xxx   或   --bearer xxx
     (注意:官方 /api/ 目前只有 images / videos,没有 music 端点,
      所以这条需要他们开一个音乐端点或给内部接口。)

  3. 批量导出(最优)
     直接要 CSV/parquet + 音频包。对我们额外的好处是能顺带要到
     **逐曲的 AI 生成标记**——Pixabay 2023 年后站内 AI 音乐占比很高,
     没有这个标记,我们自己清洗的残留污染会直接威胁实验结论(E13 教训)。

在拿到其中之一以前,这条线是堵的。人类器乐数据可以先走 MagnaTagATune
(已在 Seagate 上、有 no-vocals 标签、零外部审批)和 Jamendo API。
────────────────────────────────────────────────────────────────────
"""


def cmd_preflight(args, fetch: Fetcher):
    """只探通道,不抓数据。跑通了再跑 discover。"""
    print("[preflight] 1/3 robots.txt …")
    fetch.load_robots()

    probe = MUSIC_SEARCH.format(query="piano")
    print(f"[preflight] 2/3 robots 是否允许 {probe} …")
    print(f"           -> {'允许' if fetch.allowed(probe) else '禁止'}")

    print(f"[preflight] 3/3 实际请求 {probe} …")
    r = fetch.get(probe)
    tracks = extract_payload(r.text)
    print(f"           -> HTTP {r.status_code}, {len(r.text)} 字节, 解析出 {len(tracks)} 条候选")
    if tracks:
        print(f"           -> 样例键名: {sorted(tracks[0])[:12]}")
    dump = Path(args.out or ".") / "preflight_sample.html"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text(r.text, encoding="utf-8")
    print(f"[preflight] 原始 HTML 已落盘: {dump}")
    print("[preflight] 通道可用。可以跑 discover 了。")


def cmd_discover(args, fetch: Fetcher):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "pixabay_meta.jsonl"

    seen = set()
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["track_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[discover] 断点续跑:已有 {len(seen)} 条")

    added = 0
    with open(jsonl, "a", encoding="utf-8") as f:
        for q in args.queries:
            for page in range(1, args.pages + 1):
                url = MUSIC_SEARCH.format(query=q)
                # 只用 pagi 翻页;robots 禁止的 order/genre/duration 一概不带
                params = {"pagi": page} if page > 1 else None
                print(f"[discover] {q} p{page} …", end=" ", flush=True)
                r = fetch.get(url, params=params)
                rows = [normalize(t) for t in extract_payload(r.text)]
                rows = [x for x in rows if x["track_id"] and x["track_id"] not in seen]
                for x in rows:
                    x["query"] = q
                    seen.add(x["track_id"])
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
                f.flush()
                added += len(rows)
                print(f"+{len(rows)} (累计 {len(seen)})")
                if not rows:
                    print(f"[discover] {q} 第 {page} 页无新增,跳到下一个词")
                    break

    print(f"[discover] 完成,新增 {added} 条 → {jsonl}")
    _report(jsonl)


def _report(jsonl: Path):
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    if not rows:
        return
    ai = sum(r["ai_suspect"] for r in rows)
    inst = sum(r["instrumental_guess"] == "instrumental" for r in rows)
    voc = sum(r["instrumental_guess"] == "vocal" for r in rows)
    print(f"""
[盘点] 共 {len(rows)} 首
  AI 嫌疑(标题/标签命中生成器关键词): {ai} ({ai / len(rows):.1%})
  器乐(tag 判定): {inst}   人声: {voc}   未知: {len(rows) - inst - voc}
  注意:'未知'占比通常极高——Pixabay 没有 vocal/instrumental 字段,tag 是
  稀疏的众包正标签,「没打 vocal 标签」不等于「纯器乐」。真要用必须过一遍
  我们自己的 devocal/人声检测,不能拿这一列当真值。
""")


def cmd_fetch(args, fetch: Fetcher):
    out = Path(args.out)
    jsonl = out / "pixabay_meta.jsonl"
    if not jsonl.exists():
        sys.exit(f"没有 {jsonl},先跑 discover")
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    if args.exclude_ai:
        before = len(rows)
        rows = [r for r in rows if not r["ai_suspect"]]
        print(f"[fetch] --exclude-ai 过滤掉 {before - len(rows)} 首")
    if args.only_instrumental:
        rows = [r for r in rows if r["instrumental_guess"] != "vocal"]
    if args.limit:
        rows = rows[: args.limit]

    manifest = out / "manifest.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=list(rows[0]) + ["local_path", "sha1", "bytes"])
        if new:
            w.writeheader()
        for i, r in enumerate(rows, 1):
            dest = audio_dir / f"pixabay_{r['track_id']}.mp3"
            if dest.exists() and dest.stat().st_size > 10_000:
                continue
            if not r["audio_url"]:
                print(f"[fetch] {r['track_id']} 缺 audio_url,跳过(需要曲目页二次解析)")
                continue
            print(f"[fetch] {i}/{len(rows)} {r['track_id']} …", end=" ", flush=True)
            resp = fetch.get(r["audio_url"])
            data = resp.content
            if not any(data[: len(m)] == m for m in _MP3_MAGIC):
                print("非 mp3,跳过")
                continue
            dest.write_bytes(data)
            r2 = dict(r, local_path=str(dest), sha1=hashlib.sha1(data).hexdigest(),
                      bytes=len(data))
            w.writerow(r2)
            mf.flush()
            print(f"{len(data) // 1024} KB")
    print(f"[fetch] 完成 → {manifest}")


def main():
    ap = argparse.ArgumentParser(
        description="Pixabay 器乐人类语料采集(具名、遵守 robots、限速、不做指纹伪装)",
        epilog=BLOCKED_ADVICE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "discover", "fetch"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="")
        p.add_argument("--delay", type=float, default=2.0, help="每请求间隔秒(默认 2.0)")
        p.add_argument("--api-key", default=os.environ.get("PIXABAY_API_KEY", ""))
        p.add_argument("--bearer", default=os.environ.get("PIXABAY_BEARER", ""))
        if name == "discover":
            p.add_argument("--queries", nargs="+", required=True)
            p.add_argument("--pages", type=int, default=3)
        if name == "fetch":
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--exclude-ai", action="store_true")
            p.add_argument("--only-instrumental", action="store_true")
    args = ap.parse_args()

    contact = os.environ.get("PIXABAY_CONTACT", "")
    if not contact:
        sys.exit("请先 export PIXABAY_CONTACT=你的邮箱(会写进 User-Agent,让站方能认出我们)")
    if args.cmd in ("discover", "fetch") and not args.out:
        sys.exit("--out 必填")

    s = build_session(contact, os.environ.get("PIXABAY_EXTRA_HEADER"))
    if args.bearer:
        s.headers["Authorization"] = f"Bearer {args.bearer}"
    if args.api_key:
        s.params = {"key": args.api_key}
    fetch = Fetcher(s, delay=args.delay)

    try:
        {"preflight": cmd_preflight, "discover": cmd_discover, "fetch": cmd_fetch}[args.cmd](args, fetch)
    except AccessBlocked as e:
        print(f"\n[停机] {e}")
        print(BLOCKED_ADVICE)
        sys.exit(2)
    except RobotsDenied as e:
        print(f"\n[停机] {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
