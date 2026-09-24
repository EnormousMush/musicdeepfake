"""
建立人类侧 / AI 侧总索引(2026-09-20)。

为什么要这个:数据散在 10 个目录、schema 各不相同,两个最大的 AI 语料
(suno_audio 21,500 / fakemusiccaps 55,210)**连 manifest 都没有**。
没有统一索引就没有 split_key,没有 split_key 就无法保证切分不泄漏。

**本脚本最重要的产出不是文件清单,是 `split_key`。** 实测发现两处泄漏源:
  * `suno_audio` 的 21,500 首 = **10,750 个 prompt × 2 个 take**(uuid 完全成对)。
    随机切分会让同一 prompt 的两个 take 分到训练和测试两边。
  * `fakemusiccaps` 的 5 个生成器**共用同一批 5,521 个 MusicCaps 源 id**
    (每个 id 恰好出现在全部 5 个生成器里)。随机切分会让同一条 caption 的
    不同渲染版本跨越训练/测试,而且家族之间并不独立。
两者都会让指标**虚高**,且不会报警。

每个判断都带 `*_method` 列:「器乐」这个字在 Jamendo 是官方字段、在 MTAT 是
众包标签、在 Pixabay 是缺席推断、在 FMA 是 echonest 分数——可信度差好几档,
合成一列就把差别抹平了。

Usage:
  .venv_audit/bin/python part1_extraction/build_index.py --side ai   --out DIR
  .venv_audit/bin/python part1_extraction/build_index.py --side human --out DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
from pathlib import Path

CORP = Path("/Volumes/Seagate /honors_paper")
REAL, AIC = CORP / "1_corpora_real", CORP / "2_corpora_ai"
EXTS = (".mp3", ".wav", ".flac", ".au", ".m4a", ".ogg")

COLS = [
    "uid", "side", "corpus", "family", "path", "bytes", "sha1",
    "split_key", "split_key_method",
    "instrumental", "instrumental_method",
    "ai_risk", "ai_risk_method",
    "native_codec", "proc_history", "notes",
]


def sha1_head(p: Path, n=1 << 20):
    """前 1 MB 的 sha1。全文件哈希对 16 万首太慢,前 1MB 足以抓完全重复。

    注意:这**抓不到**「同内容不同编码」的近重复,只抓字节级重复。
    """
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def walk(d: Path):
    return [p for p in d.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith(".")]


# ------------------------------------------------------------------ AI 侧

def build_ai(do_sha):
    rows = []

    # --- suno_audio:{uuid}_{take}.mp3,uuid = prompt 组
    d = AIC / "suno_audio"
    for p in walk(d):
        m = re.match(r"(.+)_(\d+)$", p.stem)
        uuid = m.group(1) if m else p.stem
        rows.append(dict(
            corpus="suno_audio", family="suno", path=p,
            split_key=f"suno:{uuid}",
            split_key_method="文件名 uuid(实测每 uuid 恰好 2 个 take)",
            instrumental="", instrumental_method="未查",
            ai_risk="1", ai_risk_method="商用生成器，定义上是 AI",
            notes=f"genre={p.parent.name};take={m.group(2) if m else '?'}"))

    # --- fakemusiccaps:5 个生成器共用 MusicCaps 源 id
    d = AIC / "fakemusiccaps"
    for gen_dir in sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith("__")):
        for p in walk(gen_dir):
            rows.append(dict(
                corpus="fakemusiccaps", family=gen_dir.name, path=p,
                split_key=f"fmc:{gen_dir.name}:{p.stem}",
                split_key_method="文件级(源 id 另记在 notes,5 个生成器共用同一批 5,521 个)",
                instrumental="", instrumental_method="未查",
                ai_risk="1", ai_risk_method="开源 TTM 生成，定义上是 AI",
                notes=f"musiccaps_src={p.stem};同一 caption 被 5 个模型各渲染一次"))

    # --- generators:自产六家,manifest 有 caption + seed
    d = AIC / "generators"
    for batch in sorted(x for x in d.iterdir() if x.is_dir()):
        man = batch / "manifest.csv"
        meta = {}
        if man.exists():
            for r in csv.DictReader(open(man, encoding="utf-8")):
                meta[r.get("audio_id", "")] = r
        fam = batch.name.replace("_batch_1000", "")
        for p in walk(batch):
            r = meta.get(p.stem, {})
            cap = (r.get("caption", "") or "")[:60]
            key = f"{fam}:{r.get('seed','')}:{hashlib.sha1(cap.encode()).hexdigest()[:8]}" \
                  if r else f"{fam}:{p.stem}"
            rows.append(dict(
                corpus="generators", family=fam, path=p,
                split_key=f"{fam}:{p.stem}",
                split_key_method="文件级",
                instrumental="", instrumental_method="未查",
                ai_risk="1", ai_risk_method="自产，定义上是 AI",
                notes=f"genre={r.get('genre','')}"))

    # --- sonics:目录里混着三样东西,必须逐条分辨
    #   1) sunov2/v3/v35/udio* = AI
    #   2) **fma_ref = FMA 人类音乐**(当初做 demucs 对称协议留的人类参照,
    #      放在 AI 目录下但不是 AI。不分辨就会把 900 首人类曲当 AI 正样本喂进去)
    #   3) stems/ = 已过 htdemucs;sel30s/ref30s = 原始
    #      有刀/无刀混装会教坏探针——必须用 proc_history 标出来
    d = AIC / "sonics"
    fake = {}
    fp = d / "fake_songs.csv"
    if fp.exists():
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            fake[Path(r.get("filename", "")).stem] = r
    for p in walk(d):
        r = fake.get(p.stem, {})
        rel = str(p.relative_to(d))
        top = rel.split(os.sep)[0]
        bucket = rel.split(os.sep)[1] if os.sep in rel else ""
        is_human = "fma_ref" in rel
        proc = "demucs" if top == "stems" else "原始"
        rows.append(dict(
            corpus="sonics", family=("fma" if is_human else r.get("algorithm", bucket or "sonics")),
            path=p,
            split_key=f"sonics:{bucket or 'x'}:{p.parent.name}:{p.stem}",
            split_key_method="文件级(stems 与 sel30s/ref30s 是同批曲的两个版本,见 proc_history)",
            instrumental=("True" if r.get("no_vocal") == "True" else "False") if r else "",
            instrumental_method="数据集 no_vocal 列" if r else "未查",
            ai_risk=("0" if is_human else "1"),
            ai_risk_method=("**FMA 人类参照**,当初 demucs 对称协议用,误放在 AI 目录下"
                            if is_human else "商用生成器，定义上是 AI"),
            proc_history=proc,
            notes="按既定规矩只做矩阵测试行，不进 LOGO 训练池"
                  + ("；本条实为人类音乐，勿作 AI 正样本" if is_human else "")))

    # --- acestep_self
    d = AIC / "acestep_self"
    for p in walk(d):
        rows.append(dict(
            corpus="acestep_self", family="acestep", path=p,
            split_key=f"acestep_self:{p.stem}", split_key_method="文件名",
            instrumental="", instrumental_method="未查",
            ai_risk="1", ai_risk_method="自产，定义上是 AI", notes=""))
    return rows


# ---------------------------------------------------------------- 人类侧

def build_human(do_sha):
    rows = []

    def add(corpus, p, split_key, skm, inst, im, air, arm, notes=""):
        rows.append(dict(corpus=corpus, family=corpus, path=p, split_key=split_key,
                         split_key_method=skm, instrumental=inst, instrumental_method=im,
                         ai_risk=air, ai_risk_method=arm, notes=notes))

    # --- jamendo_sweep 两层:manifest 齐全
    # 2026-09-21 已物理合并:三批 jamendo 的音频统一在 jamendo/audio/,
    # manifest 里记的是旧路径,按文件名重新定位。
    JAM = REAL / "jamendo" / "audio"
    for man, sub, layer in [("manifest.csv", "audio", "main"),
                            ("manifest_audio_post2023.csv", "audio_post2023", "post2023")]:
        mp = REAL / "jamendo" / man
        if not mp.exists():
            continue
        for r in csv.DictReader(open(mp, encoding="utf-8")):
            p = JAM / Path(r["audio_path"]).name
            add("jamendo", p, f"jam:{p.stem}", "文件级",
                "True", "Jamendo musicinfo 官方字段(demucs 抽检 ~4% 明确人声)",
                r.get("ai_risk", "0"),
                "2023前结构性无AI" if layer == "main" else "行为学筛查(注册日期/上传速率/批量)",
                f"layer={layer};year={r.get('year','')}")

    # --- jamendo2025:器乐白名单单独标
    mp = REAL / "jamendo" / "manifest_jamendo.csv"
    inst_list = set()
    il = Path("part3_detector/diagnostics/jamendo_instrumental.txt")
    if il.exists():
        inst_list = {l.strip() for l in open(il) if l.strip()}
    if mp.exists():
        for r in csv.DictReader(open(mp, encoding="utf-8")):
            p = JAM / Path(r["audio_path"]).name
            is_i = r["audio_id"] in inst_list
            add("jamendo", p, f"jam:{p.stem}", "文件级",
                str(is_i), "Jamendo musicinfo 官方字段(已逐曲核查，白名单 2050/3000)",
                "0", "2024-26 发行;**未**做账号级清洗(E13 名单未找到)", f"year={r.get('releasedate','')[:4]}")

    # --- jamendo_eras
    mp = REAL / "jamendo" / "manifest_jamendo_eras.csv"
    if mp.exists():
        for r in csv.DictReader(open(mp, encoding="utf-8")):
            _p = JAM / Path(r["audio_path"]).name
            add("jamendo", _p, f"jam:{_p.stem}",
                "文件级", "True", "Jamendo musicinfo 官方字段",
                "0", "2018-22，Suno 上线前", f"year={r.get('year','')}")

    # --- pixabay:标签在隔壁 _labels,按 index 关联
    lab = REAL / "pixabay" / "labels" / "all_stage1_labeled.csv"
    strict = set()
    sp = REAL / "pixabay" / "labels" / "all_stage2_clean_strict.csv"
    if sp.exists():
        strict = {r["index"] for r in csv.DictReader(open(sp, encoding="utf-8-sig"))}
    pxmeta = {}
    if lab.exists():
        for r in csv.DictReader(open(lab, encoding="utf-8-sig")):
            pxmeta[r["index"]] = r
    for sub in ["pixabay"]:
        d = REAL / sub / "audio"
        if not d.exists():
            continue
        for p in walk(d):
            r = pxmeta.get(p.stem, {})
            tight = p.stem in strict
            add("pixabay", p, f"px:{p.stem}", "文件级",
                "True" if r.get("is_instrumental") == "True" else "",
                ("有器乐正标签 inst_tag" if tight else "仅 no_vocal 缺席推断")
                + "(demucs 抽检明确人声 1.0%/0.0%)" if r else "未查",
                "unknown",
                "平台 is_ai 仅标 0.5%、官方徽章只查到 4 首；我方打分 2.1× 基线 → 漏检严重",
                f"strict={tight}")

    # --- FMA:优先用平台流派标签 genre 1235 = "Instrumental"(14,938 首,全部在盘上),
    #     这是上传者自己打的流派,比 echonest 的算法预测值强一档。
    #     echonest instrumentalness 只覆盖 12%,降为补充判据。
    import ast
    meta = REAL / "fma" / "fma_metadata"
    fma_inst, fma_inst_top = set(), set()
    tp = meta / "tracks.csv"
    if tp.exists():
        with open(tp, encoding="utf-8") as f:
            rdr = csv.reader(f)
            hdr = [next(rdr) for _ in range(3)]
            c = {b: i for i, (a, b) in enumerate(zip(hdr[0], hdr[1])) if a == "track"}
            ga, gt = c.get("genres_all"), c.get("genre_top")
            for row in rdr:
                if not row:
                    continue
                tid = row[0].strip()
                try:
                    if ga is not None and 1235 in ast.literal_eval(row[ga] or "[]"):
                        fma_inst.add(tid)
                except (ValueError, SyntaxError):
                    pass
                if gt is not None and row[gt].strip() == "Instrumental":
                    fma_inst_top.add(tid)
    ech = {}
    ep = meta / "echonest.csv"
    if ep.exists():
        with open(ep, encoding="utf-8") as f:
            for i2, line in enumerate(f):
                if i2 < 4:
                    continue
                parts = line.split(",")
                try:
                    ech[parts[0].strip()] = float(parts[4])
                except (IndexError, ValueError):
                    pass
    for p in walk(REAL / "fma" / "fma_large"):
        tid = p.stem.lstrip("0") or "0"
        if tid in fma_inst:
            inst, im = "True", ("FMA 流派标签 Instrumental(genre 1235)"
                                + ("，且为 genre_top" if tid in fma_inst_top else ""))
        elif tid in ech:
            v = ech[tid]
            inst = "True" if v > 0.8 else "False"
            im = f"echonest instrumentalness={v:.2f}(算法预测值，非人工标注)"
        else:
            inst, im = "", "无判据(不在 Instrumental 流派，echonest 也未覆盖)"
        add("fma", p, f"fma:{p.stem}", "文件级", inst, im,
            "0", "2008-2017，生成式音乐出现前", "")

    # --- MagnaTagATune:clip_info_final.csv 是 TSV,有 artist 和 mp3_path。
    #     没有作者级切分键,同一艺术家的不同片段会跨训练/测试——模型可以靠
    #     记住这个人的音色答对,而不是识别 AI 痕迹。必须补上。
    d = REAL / "magnatagatune"
    mt_artist = {}
    ci = d / "clip_info_final.csv"
    if ci.exists():
        with open(ci, encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                mp3 = (r.get("mp3_path") or "").strip()
                if mp3:
                    mt_artist[Path(mp3).stem] = (r.get("artist") or "").strip()
    # MTAT 的 no-vocals 众包标签
    mt_novocal = set()
    ann = d / "annotations_final.csv"
    if ann.exists():
        with open(ann, encoding="utf-8", errors="replace") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            col = next((c for c in (rdr.fieldnames or []) if c.strip('"') == "no vocals"), None)
            for r in rdr:
                if col and str(r.get(col, "")).strip('"') == "1":
                    mp3 = (r.get("mp3_path") or "").strip()
                    if mp3:
                        mt_novocal.add(Path(mp3).stem)
    if d.exists():
        for p in walk(d):
            a = mt_artist.get(p.stem, "")
            add("magnatagatune", p, f"mtat:{p.stem}", "文件级",
                "True" if p.stem in mt_novocal else "",
                "annotations_final.csv 的 no-vocals 众包标签(未用 demucs 核过)",
                "0", "2000s，生成式音乐出现前", "")

    # --- fma_ref_sonics:当初 demucs 对称协议留的 FMA 人类参照,
    #     原先误放在 2_corpora_ai/sonics/ 下,2026-09-21 复制到人类侧。
    d = REAL / "fma_ref_sonics"
    if d.exists():
        for p in walk(d):
            proc = "demucs" if "stems" in str(p) else "原始"
            rows.append(dict(corpus="fma_ref_sonics", family="fma", path=p,
                             split_key=f"fma_ref:{p.parent.name}:{p.stem}",
                             split_key_method="文件级",
                             instrumental="", instrumental_method="未查",
                             ai_risk="0", ai_risk_method="FMA 人类音乐",
                             proc_history=proc,
                             notes="demucs 对称协议的人类参照;AI 侧 sonics/ 下有同一份"))

    # --- ccMixter / IA netlabels:manifest 都有 artist 列
    for sub, man in [("ccmixter", "manifest_ccmixter.csv"),
                     ("ia_netlabels", "manifest_ia.csv")]:
        d = REAL / sub
        if not d.exists():
            continue
        amap = {}
        mp = d / man
        if mp.exists():
            for r in csv.DictReader(open(mp, encoding="utf-8")):
                rel = (r.get("rel_path") or "").strip()
                if rel:
                    amap[Path(rel).stem] = (r.get("artist") or "").strip()
        for p in walk(d):
            a = amap.get(p.stem, "")
            add(sub, p, f"{sub}:{p.stem}", "文件级",
                "", "未查", "0", "年代早于生成式音乐", "")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True, choices=["human", "ai"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sha", action="store_true", help="跳过 sha1(快速试跑)")
    args = ap.parse_args()

    rows = build_ai(not args.no_sha) if args.side == "ai" else build_human(not args.no_sha)
    print(f"[{args.side}] 收集 {len(rows)} 条,开始补 sha1/大小 …", flush=True)

    out_rows, missing = [], 0
    for i, r in enumerate(rows, 1):
        p = Path(r["path"])
        if not p.exists():
            missing += 1
            continue
        r["path"] = str(p)
        r["bytes"] = p.stat().st_size
        r["sha1"] = "" if args.no_sha else sha1_head(p)
        r["side"] = args.side
        r["native_codec"] = p.suffix.lower().lstrip(".")
        r.setdefault("proc_history", "原始")
        r["uid"] = f"{r['corpus']}:{p.stem}"
        out_rows.append({k: r.get(k, "") for k in COLS})
        if i % 10000 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{args.side}_index.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n{'='*66}\n{args.side} 索引 {len(out_rows)} 行 (manifest 指向但文件不存在 {missing})")
    c = collections.Counter(r["corpus"] for r in out_rows)
    for k, v in c.most_common():
        g = len({r["split_key"] for r in out_rows if r["corpus"] == k})
        print(f"  {k:16s} {v:7d} 首   {g:6d} 个切分组")
    mis = [r for r in out_rows if r["side"] == "ai" and r["ai_risk"] == "0"]
    if mis:
        print(f"\n  ⚠️ AI 目录下实为人类音乐: {len(mis)} 首(已标 ai_risk=0,勿作正样本)")
    pc = collections.Counter(r["proc_history"] for r in out_rows)
    print(f"  处理历史: {dict(pc)}")
    if not args.no_sha:
        h = collections.Counter(r["sha1"] for r in out_rows if r["sha1"])
        dup = {k: v for k, v in h.items() if v > 1}
        ndup = sum(v - 1 for v in dup.values())
        print(f"\n  字节级重复: {len(dup)} 组, 多余 {ndup} 首")
    print(f"→ {f}")


if __name__ == "__main__":
    main()
