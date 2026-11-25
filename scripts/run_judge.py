"""对一局/一批发言做 LLM 质性编码。

用法：
  python scripts/run_judge.py --session <sid> --model <judge_model_id> [--limit 12]
  python scripts/run_judge.py --session <sid> --model <judge_model_id> \
      --rubric rubrics/moral_gaming.json

评分表（--rubric）两种形状都接受：
  单个 rubric：  [{"id": "cooperation", "q": "...", "range": [1, 5]}, ...]
  分批 rubric：  [[...], [...], [...]]   ← 按维度分批打分，推荐

为什么推荐分批：一次让裁判打 30 个字段，注意力会被摊薄并互相污染
（"看起来不诚实"会顺手压低"论证质量"）。按 范围 / 判准 / 硬约束 / 论证质量
分成四批，每批只问一族问题，信度明显更高；每批还能有自己的取值范围。

默认不截断发言：论证结构（理由链、让步、对反驳的回应）常在长发言的后半段，
截断会让数据看着完整、其实失真。要省 token 时用 --max-chars 显式指定。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "server"))

import base.judge as judge_mod      # noqa: E402
import store                        # noqa: E402


def load_session_items(sid: str, limit: int):
    path = os.path.join(ROOT, "runs", "sessions", sid + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit("找不到日志：" + path)
    texts = {}
    order = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "message_start":
                m = ev.get("message") or {}
                if m.get("sender") is None:
                    continue
                texts[m.get("id")] = {"seat": m.get("sender"), "text": ""}
                order.append(m.get("id"))
            elif ev.get("type") == "message_end":
                if ev.get("id") in texts:
                    texts[ev["id"]]["text"] = ev.get("text") or ""
    items = [texts[i] for i in order if (texts[i].get("text") or "").strip()]
    return items[-limit:] if limit else items


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8-sig") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise SystemExit("找不到评分表：" + path)
    except json.JSONDecodeError as exc:
        raise SystemExit("评分表不是合法 JSON（{}）：{}".format(exc, path))


def _as_batches(data, path: str, key: str = "", require: bool = True) -> list:
    """把一段 JSON 规范成"批次列表"：单批也包成一批，调用方只有一条路径。"""
    if isinstance(data, dict):
        data = data.get(key) or []
    if not isinstance(data, list):
        raise SystemExit("评分表要是一个数组：" + path)
    if not data:
        if require:
            raise SystemExit("评分表要是一个非空数组：" + path)
        return []
    if all(isinstance(x, dict) for x in data):
        return [data]
    if all(isinstance(x, list) for x in data):
        return [batch for batch in data if batch]
    raise SystemExit("评分表格式混用：要么全是评分项，要么全是批次：" + path)


def load_rubric(path: str) -> list:
    """逐条发言用的批次列表。

    兼容三种形状：单个评分项数组、批次数组、以及 {turn, seat} 结构（取 turn）。
    """
    return _as_batches(_read_json(path), path, "turn")


def load_rubric_book(path: str) -> dict:
    """返回 {"turn": [...], "seat": [...]}。

    seat 那批不是逐条判的：它把同一个座位的全部发言拼成一条再判。
    因为"隐瞒"是整场属性——单看一轮，永远说不出这个人是不是始终没披露。
    """
    data = _read_json(path)
    if isinstance(data, dict):
        return {"turn": _as_batches(data, path, "turn"),
                "seat": _as_batches(data, path, "seat", require=False)}
    return {"turn": _as_batches(data, path), "seat": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--context", default="")
    ap.add_argument("--rubric", default="",
                    help="JSON 评分表；给分批形式则按维度分批打分")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="每条发言截断到多少字符；0（默认）= 不截断")
    args = ap.parse_args()

    conf = store.get_model(args.model, with_secret=True)
    if not conf:
        raise SystemExit("模型不存在：" + args.model)
    items = load_session_items(args.session, args.limit)
    if not items:
        raise SystemExit("这局没有可评的发言")

    rubrics = load_rubric(args.rubric) if args.rubric else None
    max_chars = args.max_chars or None
    context = args.context or args.session

    print("评 {} 条发言，judge = {}，评分批次数 = {}".format(
        len(items), args.model, len(rubrics) if rubrics else 1))

    if rubrics:
        result = judge_mod.judge_batches(conf, items, rubrics, context=context,
                                         max_chars=max_chars)
        if result.get("failed_batches"):
            print("警告：第 {} 批没拿到分，这些字段会是空的".format(
                "、".join(str(b) for b in result["failed_batches"])))
    else:
        result = judge_mod.judge(conf, items, context=context, max_chars=max_chars)

    path = judge_mod.save(result, args.session)
    print("聚合：", json.dumps(result["aggregate"], ensure_ascii=False))
    for s in result["scores"][:5]:
        print("  [{}] {}号 {} {}".format(s.get("index"), s.get("seat"), s.get("scores"),
                                        (s.get("note") or "")[:40]))
    print("结果已存：", path)


if __name__ == "__main__":
    main()
