"""批量实验 CLI。

用法：
  python scripts/run_batch.py --plugin werewolf --scenario scenarios/werewolf12_standard.yaml \
      --seats 12 --seeds 1-3 --sweep "communication=on,off" --resume

座位轮换（模型 × 角色正交）：
  python scripts/run_batch.py --plugin moral_gaming --seats 3 \
      --sweep "assign=L1,L2,L3;order=O1,O2,O3,O4,O5,O6" --seeds 1-10

  assign 决定"哪个模型坐哪个座位"，order 由插件自己解释（发言顺序）。
  两个键相乘即为条件矩阵；先加 --check-only 看一眼展开对不对，再真跑。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "server"))

import store                                   # noqa: E402
from base import batch as batch_mod            # noqa: E402
from base import metrics as metrics_mod         # noqa: E402

# 历史遗留：曾有一个不该参与实验的注册项，用 id 硬过滤掉。
# 保留过滤，但命中时打印出来——否则实验会以"少一家"的阵容跑完，
# 而报告里看不出任何异常。要强制使用请用 --seat-models 显式指定。
LEGACY_EXCLUDED_MODEL_IDS = ("model-d3cf3bfe",)


def parse_seeds(text: str):
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_sweep(text: str):
    sweep = {}
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, vals = chunk.partition("=")
        items = []
        for v in vals.split(","):
            v = v.strip()
            if not v:
                continue
            if v in ("on", "off"):
                items.append(v)
            elif v.lower() in ("true", "false"):
                items.append(v.lower() == "true")
            else:
                try:
                    items.append(int(v))
                except ValueError:
                    try:
                        items.append(float(v))
                    except ValueError:
                        items.append(v)
        sweep[key.strip()] = items
    return sweep


def resolve_seats(count: int, model_ids: str = ""):
    """按模型库顺序取座位；给了 model_ids 就严格按这个顺序取。

    顺序即座位号：第 1 个模型坐 1 号位。轮换交给 batch 的 assign 条件完成，
    这里只决定基准顺序。
    """
    wanted = [s.strip() for s in (model_ids or "").split(",") if s.strip()]
    all_models = list(store.list_models())
    by_id = {m["id"]: m for m in all_models}

    if wanted:
        missing = [w for w in wanted if w not in by_id]
        if missing:
            raise SystemExit("模型库里找不到这些 id：{}".format(", ".join(missing)))
        picked = [by_id[w] for w in wanted]
    else:
        skipped = [m["id"] for m in all_models if m["id"] in LEGACY_EXCLUDED_MODEL_IDS]
        if skipped:
            print("提示：按历史遗留名单跳过 {}（要强制使用请加 --seat-models）".format(
                ", ".join(skipped)))
        picked = [m for m in all_models
                  if m["id"] not in LEGACY_EXCLUDED_MODEL_IDS][:count]

    seats = []
    for i, m in enumerate(picked, start=1):
        conf = store.get_model(m["id"], with_secret=True)
        seats.append({"name": "{}号".format(i), "model_conf": conf})
    return seats


def describe_assigns(seats, sweep) -> None:
    """打印每个 assign 条件下谁坐哪个位子：这是最容易被默认错的地方。"""
    for value in (sweep or {}).get("assign") or []:
        try:
            mapping = batch_mod.seat_model_map(batch_mod.rotate_seats(seats, value))
        except ValueError as exc:
            print("  assign={:<3} 无效：{}".format(value, exc))
            continue
        print("  assign={:<3} 座位→模型  {}".format(
            value, "  ".join("{}:{}".format(k, v) for k, v in mapping.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="werewolf")
    ap.add_argument("--scenario", default="scenarios/werewolf12_standard.yaml")
    ap.add_argument("--seats", type=int, default=12)
    ap.add_argument("--seat-models", default="",
                    help="逗号分隔的模型 id，按顺序坐 1..N 号位；留空则按模型库顺序")
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--check-only", action="store_true",
                    help="只展开条件矩阵并打印座位映射，不调用任何模型")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="同时跑几局；默认 1（串行）。>1 必须同时给 --allow-parallel")
    ap.add_argument("--allow-parallel", action="store_true",
                    help="显式授权并发（未经允许不要用）")
    args = ap.parse_args()

    seats = resolve_seats(args.seats, args.seat_models)
    if not seats:
        print("模型库为空，先在模型管理页添加模型")
        return 2
    # 不同插件用不同键：狼人杀看 rounds_per_day，公地/规则怪谈看 rounds
    base_config = {"scenario": args.scenario, "rounds_per_day": args.rounds,
                   "rounds": args.rounds}
    try:
        concurrency = batch_mod.check_concurrency(args.concurrency, args.allow_parallel)
    except ValueError as exc:
        print(str(exc))
        return 2
    sweep = parse_sweep(args.sweep)
    seeds = parse_seeds(args.seeds)

    if args.check_only:
        # 只展开、不调用模型：先确认 18 个格子真的是 18 个、轮换真的是正交的
        try:
            batch_mod.validate_assigns(sweep, len(seats))
        except ValueError as exc:
            print("条件有误：{}".format(exc))
            return 2
        specs = batch_mod.plan_runs("dryrun", sweep, seeds)
        print("插件 {} | 座位 {} | 条件 {} | 将执行 {} 局".format(
            args.plugin, len(seats), sweep or "（单条件）", len(specs)))
        for spec in specs:
            print("  {}".format(spec.run_id))
        describe_assigns(seats, sweep)
        return 0

    print("插件 {} | 座位 {} | seeds {} | 条件 {} | 并发 {}".format(
        args.plugin, len(seats), seeds, sweep or "（单条件）", concurrency))
    describe_assigns(seats, sweep)

    def progress(row, done, total):
        print("  [{}/{}] {} | {} | winner={} | {}s".format(
            done, total, row["run_id"], row.get("state"), row.get("winner", ""),
            row.get("duration_s")))

    summary = batch_mod.run_batch(
        args.plugin, seats=seats, base_config=base_config, sweep=sweep, seeds=seeds,
        title=args.plugin, out_root=args.out or None,
        resume=not args.no_resume, concurrency=concurrency,
        on_progress=progress)
    print("\n批次完成:", summary["batch_id"])
    print("  索引:", summary["index"])
    print("  清单:", summary["manifest"])
    if summary.get("report"):
        print("  报告:", summary["report"])
    print("  执行 {} 局，跳过 {} 局，错误 {} 个".format(
        summary["executed"], summary["skipped"], len(summary["errors"])))

    # 聚合
    import csv
    rows = list(csv.DictReader(open(summary["index"], encoding="utf-8-sig")))
    def _is_num(key):
        if key == "duration_s":
            return True
        if not key.startswith("m_"):
            return False
        try:
            float(rows[0].get(key))
            return True
        except (TypeError, ValueError):
            return False

    numeric = [k for k in (rows[0].keys() if rows else []) if _is_num(k)]
    agg = metrics_mod.aggregate(rows, numeric)
    print("\n聚合（n={}）:".format(agg["n"]))
    for k in numeric:
        v = agg.get(k) or {}
        if v.get("mean") is not None:
            print("  {:<20} mean={:<10} sd={:<10} ci95=±{}".format(
                k, v["mean"], v["sd"], v["ci95"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
