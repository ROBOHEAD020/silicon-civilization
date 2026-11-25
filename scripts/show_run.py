"""把一局的会话日志（JSONL）渲染成可读的对话记录。

用法：
    python scripts/show_run.py <run_id>                 # 从 runs/sessions 里找
    python scripts/show_run.py runs/sessions/xxx.jsonl
    python scripts/show_run.py <run_id> --thinking      # 连思考一起打
    python scripts/show_run.py <run_id> --raw           # 原样打 JSON

为什么需要它：日志是事件流，一条发言被拆成 message_start（占位）和
message_end（终稿），中间还夹着 delta 增量。直接看 JSONL 读不出来。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve(target: str) -> Path:
    p = Path(target)
    if p.exists():
        return p
    p2 = ROOT / "runs" / "sessions" / (target + ".jsonl")
    if p2.exists():
        return p2
    raise SystemExit("找不到日志：" + target)


def render(path: Path, with_thinking: bool = False) -> None:
    pending: dict = {}
    names: dict = {}
    title = ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            kind = ev.get("type")
            if kind == "meta":
                title = ev.get("title") or ""
                names = {a.get("id"): a.get("name") for a in ev.get("agents") or []}
            elif kind == "round_start":
                rnd = ev.get("round") or {}
                print("\n[{}]{}".format(rnd.get("title", ""),
                                        "  " + rnd["subtitle"] if rnd.get("subtitle") else ""))
            elif kind == "round_end":
                if ev.get("summary"):
                    print("  ↳ " + str(ev["summary"]))
            elif kind == "message_start":
                m = ev.get("message") or {}
                if m.get("kind") == "referee" and (m.get("text") or "").strip():
                    print("  ▸ " + m["text"].strip())
                elif m.get("kind") in ("speak", "action"):
                    pending[m.get("id")] = m
            elif kind == "message_end":
                m = pending.pop(ev.get("id"), None)
                if m is None:
                    continue
                text = (ev.get("text") or "").strip()
                if not text:
                    continue
                who = names.get(m.get("sender")) or "{}号".format(m.get("sender"))
                tail = "  〔{}〕".format(m["channel"]) if m.get("channel") not in (None, "public") else ""
                print("\n{}：{}".format(who, tail))
                print("  " + text.replace("\n", "\n  "))
                if with_thinking and (ev.get("thinking") or "").strip():
                    print("  （思考）" + ev["thinking"].strip().replace("\n", " ")[:300])
            elif kind == "session" and ev.get("state") == "ended":
                print("\n=== 结束：winner={} ===".format(ev.get("winner")))
    print("\n—— {} ——".format(title or path.name))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("用法：python scripts/show_run.py <run_id 或 jsonl 路径> [--thinking]")
    path = resolve(args[0])
    if "--raw" in sys.argv:
        print(path.read_text(encoding="utf-8", errors="replace"))
        return 0
    render(path, with_thinking="--thinking" in sys.argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
