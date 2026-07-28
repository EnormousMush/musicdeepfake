"""
跨生成器测试 · Mureka(商用 API)instrumental 批量提取 — 同冻结 prompt。

商用第三家(继 Suno / SONICS-Udio 之后)。BGM 端点 $0.045/首(V8/V9)、$0.03(V7.6);
一次生成默认出 2 首(计费按首),尝试 n=1 控制;串行(充值档 1 并发)。

API key 从环境变量 MUREKA_API_KEY 读取(写进 ~/.zshrc,不进 git、不进聊天):
  python mureka_generate.py --prompts suno_prompts_all.json --out mureka_batch
  干跑:--limit 10(先跑 10 首,对账单核实单价再放量)
"""
import argparse
import csv
import os
import time
from pathlib import Path

import requests

from crossgen_prompts import sample_plan

API = "https://api.mureka.ai"


def generate_one(key, model, prompt, n, timeout_s=300):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt}
    if n:
        payload["n"] = n
    r = requests.post(f"{API}/v1/instrumental/generate", json=payload,
                      headers=headers, timeout=60)
    r.raise_for_status()
    task_id = r.json().get("id")
    if not task_id:
        raise RuntimeError(f"无 task id: {r.text[:200]}")
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"task {task_id} 超时")
        q = requests.get(f"{API}/v1/instrumental/query/{task_id}",
                         headers=headers, timeout=60)
        q.raise_for_status()
        res = q.json()
        status = res.get("status")
        if status == "succeeded":
            urls = [c.get("url") for c in res.get("choices", []) if c.get("url")]
            if not urls:
                raise RuntimeError(f"succeeded 但无 url: {res}")
            return urls, res.get("model", model)
        if status in ("failed", "cancelled", "timeouted"):
            raise RuntimeError(f"task {task_id}: {status}")
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="suno_prompts_all.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="auto", help="auto=最新(V9 档 $0.045/首)")
    ap.add_argument("--n", type=int, default=1, help="每次生成几首;API 不认就报错回退")
    ap.add_argument("--n-per-genre", type=int, default=125)
    ap.add_argument("--limit", type=int, default=None, help="干跑:总共只生成 N 首")
    ap.add_argument("--keep-all-takes", action="store_true",
                    help="n>1 或 API 强制多首时,全部保留(_take2 后缀)")
    args = ap.parse_args()

    key = os.environ.get("MUREKA_API_KEY")
    if not key:
        raise SystemExit("请先 export MUREKA_API_KEY=...(不要写进代码或聊天)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.csv"

    plan = sample_plan(args.prompts, prefix="mureka", n_per_genre=args.n_per_genre)
    if args.limit:
        plan = plan[: args.limit]
    done = set()
    if man_path.exists():
        done = {r["audio_id"] for r in csv.DictReader(open(man_path))}
    todo = [p for p in plan if p["audio_id"] not in done]
    print(f"计划 {len(plan)} 首 | 已完成 {len(plan)-len(todo)} | 待生成 {len(todo)}", flush=True)
    if not todo:
        return

    new_manifest = not man_path.exists()
    mf = open(man_path, "a", newline="")
    w = csv.writer(mf)
    if new_manifest:
        w.writerow(["audio_id", "genre", "caption", "seed", "rel_path", "model",
                    "n_choices", "gen_time_s"])
    ok, fail, t0 = 0, 0, time.time()
    for i, p in enumerate(todo, 1):
        t1 = time.time()
        try:
            urls, used_model = generate_one(key, args.model, p["caption"], args.n)
            keep = urls if args.keep_all_takes else urls[:1]
            rels = []
            for j, u in enumerate(keep):
                ext = Path(u.split("?")[0]).suffix or ".mp3"
                name = p["audio_id"] + (f"_take{j+1}" if j else "") + ext
                data = requests.get(u, timeout=120)
                data.raise_for_status()
                (out / name).write_bytes(data.content)
                rels.append(name)
            w.writerow([p["audio_id"], p["genre"], p["caption"], p["seed"],
                        ";".join(rels), used_model, len(urls),
                        round(time.time() - t1, 1)])
            mf.flush(); ok += 1
        except Exception as e:
            print(f"  ❌ {p['audio_id']}: {e!r}", flush=True); fail += 1
            if fail >= 5 and ok == 0:
                print("连败 5 次,疑似参数/余额问题,停止。", flush=True); break
        if i % 10 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            eta = rate * (len(todo) - i) / 60
            print(f"  {i}/{len(todo)} 完成 ({rate:.0f}s/首, 预计还需 {eta:.0f} 分钟, 失败 {fail})",
                  flush=True)
    mf.close()
    print(f"\n完成:成功 {ok},失败 {fail} -> {out}")


if __name__ == "__main__":
    main()
