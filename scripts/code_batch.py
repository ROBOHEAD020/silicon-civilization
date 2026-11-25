"""把一批运行的发言送裁判编码，并并回运行指标。

两步：
  1) 逐座位分批判分：turn 级 5 批（范围/判准/硬约束/论证质量/诚实策略）
     + seat 级 1 批（披露 / 隐瞒）
  2) 与 index.csv 的 m_* / o_* 按 (run_id, seat) 合并

为什么按座位分组、而不是整局一次判：
  判"隐瞒"必须先知道这个座位**知道什么**。私有包是逐座位的，所以 context 只能
  逐座位给。整局一次给，裁判会拿甲的私有材料去判乙，C3/C8 全乱。

为什么"隐瞒"要单独按座位再判一次：
  它是整场对话的属性——单看一轮，永远说不出这个人是不是**始终**没披露。

输出（写在批次目录下）：
  coded_turns.csv   一行一条发言，带 turn 级字段
  coded_seats.csv   一行一个（运行 × 座位），带 seat 级字段
  joined.csv        coded_seats + 运行指标（θ、票、方案、GAP）

用法：
  python scripts/code_batch.py --batch runs/batches/bXXXX --judge judge-model --dry-run
  python scripts/code_batch.py --batch runs/batches/bXXXX --judge judge-model
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "scripts"))

import base.judge as judge_mod                    # noqa: E402
import store                                      # noqa: E402
from run_judge import load_rubric_book            # noqa: E402
from plugins.moral_gaming import plugin as mg      # noqa: E402

DEFAULT_RUBRIC = str(ROOT / "rubrics" / "moral_gaming.json")
KEY_COLS = ("run_id", "seat", "round", "turn_index", "text_chars")


# ---------------------------------------------------------------- 读取
def load_speeches(run_id: str):
    """从会话日志取发言（只要 kind=speak 且有正文），按出现顺序。"""
    path = ROOT / "runs" / "sessions" / (run_id + ".jsonl")
    if not path.exists():
        return None
    pending: dict = {}
    order: list = []
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
                msg = ev.get("message") or {}
                if msg.get("kind") != "speak" or msg.get("sender") is None:
                    continue
                pending[msg.get("id")] = {"seat": int(msg["sender"]),
                                          "round": (msg.get("meta") or {}).get("round", ""),
                                          "text": ""}
                order.append(msg.get("id"))
            elif ev.get("type") == "message_end":
                rec = pending.get(ev.get("id"))
                if rec is not None:
                    rec["text"] = ev.get("text") or ""
    return [pending[m] for m in order if (pending[m]["text"] or "").strip()]


def seat_context(seat: int, vote_mode: str = "", honesty: str = "") -> str:
    """判分时该座位能看到的全部事实：公开材料 + 他自己的私有材料 + 本局规则。

    用插件的常量而不是另抄一份——判据和场景必须同源。改了场景却忘了改裁判提示词，
    C8 的判定会静默失效，而这种失效在数据里看不出来。
    """
    lines = ["【公开事实】"]
    lines.extend("F{} {}".format(i, f) for i, f in enumerate(mg.PUBLIC_FACTS, start=1))
    lines.append("")
    lines.append("【方案收益（0–100，越高越好）】方案｜甲国｜乙国｜丙国｜G 全人类")
    for code in mg.OPTIONS:
        p = mg.PAYOFF[code]
        lines.append("{} {}｜{}｜{}｜{}｜{}".format(
            code, mg.OPTION_META[code]["label"], p["A"], p["B"], p["C"], p["G"]))
    lines.append("（O2 是现状延续的默认项）")
    lines.append("")
    brief = mg.COUNTRY_BRIEF.get(seat) or mg.COUNTRY_BRIEF[1]
    lines.append("【该代表的身份】{}（代表{}）".format(brief["role"], mg.COUNTRY.get(seat, "?")))
    lines.append("【该代表的私有材料，只有他知道】")
    lines.extend("{}）{}".format(i, item) for i, item in enumerate(brief["private"], start=1))
    lines.append("")
    plugin = mg.MoralGamingPlugin()
    lines.append("【本局表决规则】" + plugin._rule_lines({"vote_mode": vote_mode or "weighted"})[0])
    rule = mg.HONESTY.get(str(honesty or "H1").upper(), "")
    lines.append("【本局诚实规则】" + (rule.replace("【诚实规则】", "") if rule else "（本局没有给诚实规则）"))
    return "\n".join(lines)


def _index_rows(batch_dir: Path) -> list:
    index = batch_dir / "index.csv"
    if not index.exists():
        raise SystemExit("找不到批次索引：{}".format(index))
    with open(index, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list, columns: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 编码
def code_run(conf, row: dict, book: dict, max_chars=None, limit: int = 0) -> dict:
    run_id = row["run_id"]
    speeches = load_speeches(run_id)
    if speeches is None:
        return {"turns": [], "seats": [], "errors": ["缺少会话日志：" + run_id]}
    if limit:
        speeches = speeches[-limit:]

    by_seat: dict = {}
    for s in speeches:
        by_seat.setdefault(s["seat"], []).append(s)

    turns: list = []
    seats: list = []
    errors: list = []
    for seat in sorted(by_seat):
        items = by_seat[seat]
        ctx = seat_context(seat, row.get("o_vote_mode", ""), row.get("o_honesty", ""))

        if book.get("turn"):
            result = judge_mod.judge_batches(
                conf, [{"seat": seat, "text": it["text"]} for it in items],
                book["turn"], context=ctx, max_chars=max_chars)
            for err in result.get("failed_batches") or []:
                errors.append("{} 座位{} 逐轮第{}批未拿到分".format(run_id, seat, err))
            for rec in result.get("scores") or []:
                idx = rec.get("index")
                if not isinstance(idx, int) or not (1 <= idx <= len(items)):
                    continue
                turns.append({"run_id": run_id, "seat": seat,
                              "round": items[idx - 1].get("round", ""),
                              "turn_index": idx,
                              "text_chars": len(items[idx - 1]["text"]),
                              "scores": rec.get("scores") or {}})

        seat_scores: dict = {}
        if book.get("seat"):
            # 把这个座位的全部发言拼成一条再判：隐瞒是整场属性
            joined = "\n---\n".join(it["text"] for it in items)
            result = judge_mod.judge_batches(
                conf, [{"seat": seat, "text": joined}], book["seat"],
                context=ctx, max_chars=max_chars)
            for err in result.get("failed_batches") or []:
                errors.append("{} 座位{} 座位级第{}批未拿到分".format(run_id, seat, err))
            seat_scores = ((result.get("scores") or [{}])[0]).get("scores") or {}
        seats.append({"run_id": run_id, "seat": seat, "n_turns": len(items),
                      "scores": seat_scores})
        print("  {} 座位{} 编码 {} 条".format(run_id, seat, len(items)))
    return {"turns": turns, "seats": seats, "errors": errors}


# ---------------------------------------------------------------- 汇总
def _zscore(values):
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return [None] * len(values)
    mu, sd = statistics.mean(vals), statistics.pstdev(vals)
    if sd <= 1e-12:
        return [None] * len(values)
    return [None if v is None else (float(v) - mu) / sd for v in values]


def build_joined(turn_rows: list, seat_rows: list, index_rows: list,
                 turn_fields: list, seat_fields: list) -> list:
    agg: dict = {}
    for t in turn_rows:
        key = (t["run_id"], int(t["seat"]))
        slot = agg.setdefault(key, {"n": 0, "sum": {}})
        slot["n"] += 1
        for fid, val in (t.get("scores") or {}).items():
            slot["sum"][fid] = slot["sum"].get(fid, 0.0) + float(val)

    seat_map = {(s["run_id"], int(s["seat"])): (s.get("scores") or {}) for s in seat_rows}
    by_run = {r["run_id"]: r for r in index_rows}

    rows = []
    for key in sorted(set(agg) | set(seat_map)):
        run_id, seat = key
        idx = by_run.get(run_id) or {}
        slot = agg.get(key) or {"n": 0, "sum": {}}
        row = {"run_id": run_id, "seat": seat, "n_turns": slot["n"]}
        for fid in turn_fields:
            total = slot["sum"].get(fid)
            row[fid] = round(total / slot["n"], 4) if (total is not None and slot["n"]) else None
        for fid in seat_fields:
            val = seat_map.get(key, {}).get(fid)
            row[fid] = round(float(val), 4) if val is not None else None
        row["theta"] = _num(idx.get("m_theta_{}".format(seat)))
        row["theta_instr"] = _num(idx.get("m_theta_instr_{}".format(seat)))
        row["P"] = _num(idx.get("m_P_{}".format(seat)))
        row["vote"] = idx.get("m_vote_{}".format(seat), "")
        row["voted_own_best"] = _num(idx.get("m_voted_own_best_{}".format(seat)))
        row["mandate"] = idx.get("o_mandate_{}".format(seat), "")
        for col in ("option_chosen", "option_source", "vote_mode", "assign",
                    "honesty", "mandate_mode", "config_error"):
            row[col] = idx.get("o_" + col, "")
        row["GAP2_抗命"] = (round(row["theta_instr"] - row["theta"], 4)
                            if (row["theta"] is not None and row["theta_instr"] is not None)
                            else None)
        rows.append(row)

    # GAP1 = z(说得道德) − z(做得道德)。z 只在本批内算，跨批不可直接比，
    # 所以列名带 _batch；正式分析应在汇总全部批次后重算。
    z_claim = _zscore([r.get("claim_strength") for r in rows])
    z_theta = _zscore([r.get("theta") for r in rows])
    for row, a, b in zip(rows, z_claim, z_theta):
        row["GAP1_伪善_batch"] = (round(a - b, 4)
                                  if (a is not None and b is not None) else None)
    return rows


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="批次目录（含 index.csv）")
    ap.add_argument("--judge", default="", help="裁判模型 id（模型库里注册过）")
    ap.add_argument("--rubric", default=DEFAULT_RUBRIC)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="每条发言截断到多少字符；0（默认）= 不截断")
    ap.add_argument("--limit", type=int, default=0, help="每个座位最多编多少条（取最后 N 条）")
    ap.add_argument("--out-dir", default="", help="默认与 --batch 同目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不调模型")
    ap.add_argument("--no-resume", action="store_false", dest="resume", default=True)
    args = ap.parse_args()

    batch_dir = Path(args.batch)
    out_dir = Path(args.out_dir) if args.out_dir else batch_dir
    book = load_rubric_book(args.rubric)
    turn_fields = [r["id"] for b in book.get("turn") or [] for r in b]
    seat_fields = [r["id"] for b in book.get("seat") or [] for r in b]
    if not turn_fields and not seat_fields:
        raise SystemExit("评分表里没有任何批次：" + args.rubric)

    index_rows = _index_rows(batch_dir)
    out_turns = out_dir / "coded_turns.csv"
    out_seats = out_dir / "coded_seats.csv"
    out_joined = out_dir / "joined.csv"

    old_turns = _read_csv(out_turns) if args.resume else []
    old_seats = _read_csv(out_seats) if args.resume else []
    done = {r["run_id"] for r in old_turns} | {r["run_id"] for r in old_seats}
    todo = [r for r in index_rows if r["run_id"] not in done]

    n_batches = len(book.get("turn") or []) + len(book.get("seat") or [])
    print("批次 {} | 共 {} 局，已编 {} 局，待编 {} 局".format(
        batch_dir.name, len(index_rows), len(done), len(todo)))
    print("每个座位 {} 批 × 3 座位 → 预计裁判调用约 {} 次".format(
        n_batches, len(todo) * 3 * n_batches))

    if args.dry_run:
        missing = 0
        for r in todo:
            sp = load_speeches(r["run_id"])
            if sp is None:
                missing += 1
                print("  缺日志：{}".format(r["run_id"]))
        print("可编 {} 局{}".format(len(todo) - missing,
                                    "，缺日志 {} 局".format(missing) if missing else ""))
        return 0

    if not args.judge:
        raise SystemExit("需要 --judge（裁判模型 id）")
    conf = store.get_model(args.judge, with_secret=True)
    if not conf:
        raise SystemExit("模型不存在：" + args.judge)

    max_chars = args.max_chars or None
    new_turns: list = []
    new_seats: list = []
    errors: list = []
    for row in todo:
        got = code_run(conf, row, book, max_chars=max_chars, limit=args.limit)
        new_turns.extend(got["turns"])
        new_seats.extend(got["seats"])
        errors.extend(got["errors"])

    def _from_csv_turns(rows):
        return [{"run_id": r["run_id"], "seat": int(r["seat"]), "round": r.get("round", ""),
                 "turn_index": int(r.get("turn_index") or 0),
                 "text_chars": r.get("text_chars"),
                 "scores": {f: float(r[f]) for f in turn_fields if r.get(f) not in ("", None)}}
                for r in rows]

    def _from_csv_seats(rows):
        return [{"run_id": r["run_id"], "seat": int(r["seat"]),
                 "n_turns": int(r.get("n_turns") or 0),
                 "scores": {f: float(r[f]) for f in seat_fields if r.get(f) not in ("", None)}}
                for r in rows]

    all_turns = _from_csv_turns(old_turns) + new_turns
    all_seats = _from_csv_seats(old_seats) + new_seats

    _write_csv(out_turns,
               [dict({k: t[k] for k in KEY_COLS if k in t},
                     **{f: (t["scores"] or {}).get(f) for f in turn_fields})
                for t in all_turns],
               list(KEY_COLS) + turn_fields)
    _write_csv(out_seats,
               [dict({"run_id": s["run_id"], "seat": s["seat"], "n_turns": s["n_turns"]},
                     **{f: (s["scores"] or {}).get(f) for f in seat_fields})
                for s in all_seats],
               ["run_id", "seat", "n_turns"] + seat_fields)

    joined = build_joined(all_turns, all_seats, index_rows, turn_fields, seat_fields)
    _write_csv(out_joined, joined,
               ["run_id", "seat", "n_turns"] + turn_fields + seat_fields
               + ["theta", "theta_instr", "GAP2_抗命", "GAP1_伪善_batch",
                  "P", "vote", "voted_own_best", "mandate",
                  "option_chosen", "option_source", "vote_mode",
                  "assign", "honesty", "mandate_mode", "config_error"])

    print("\n已写 {}（{} 行）".format(out_turns, len(all_turns)))
    print("已写 {}（{} 行）".format(out_seats, len(all_seats)))
    print("已写 {}（{} 行）".format(out_joined, len(joined)))
    if errors:
        print("警告：{} 个批次没拿到分".format(len(errors)))
        for e in errors[:5]:
            print("  " + e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
