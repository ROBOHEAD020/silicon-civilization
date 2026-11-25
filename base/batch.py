"""批量实验运行器（与场景无关）。

一次批次 = 条件矩阵 × 随机种子 × 重复；支持并发与断点续跑。
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import base.metrics as metrics_mod          # noqa: E402
import base.manifest as manifest_mod         # noqa: E402
from base import plugin as plugin_mod        # noqa: E402
from base import runner as runner_mod        # noqa: E402


@dataclass
class RunSpec:
    run_id: str
    condition: dict = field(default_factory=dict)
    seed: int = 0

    @property
    def config(self) -> dict:
        return dict(self.condition)


def check_concurrency(value, allow_parallel: bool = False) -> int:
    """并发闸门：没有显式授权时，最多只允许 1 局同时跑。

    用户在 2026-10-28 明确要求：未经允许不得发起大规模并发测试。
    因此这里默认拒绝 >1，必须显式传 allow_parallel=True（CLI 的 --allow-parallel）。
    """
    try:
        n = int(value or 1)
    except (TypeError, ValueError):
        n = 1
    if n <= 1:
        return 1
    if not allow_parallel:
        raise ValueError(
            "并发 {} 未获授权：默认只允许单局串行。".format(n) +
            "如需并发，请显式加 --allow-parallel（或界面勾选授权），并先确认额度与限流。")
    return n


def build_conditions(sweep: dict) -> list:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    values = [[v] if not isinstance(v, (list, tuple)) else list(v) for v in
              (sweep[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def condition_name(cond: dict) -> str:
    if not cond:
        return "base"
    parts = []
    for k, v in cond.items():
        s = str(v).replace(".", "p").replace("-", "n")
        parts.append("{}-{}".format(k, s))
    return "_".join(parts)


# ---------- 座位 ↔ 模型轮换（拉丁方） ----------
# 如果每个批次都用同一个"座位 → 模型"对应关系，那么"模型差异"与"角色差异"
# 就完全混淆：V1 永远演 1 号位，观测到的差别既可能是模型价值观，也可能只是
# 这个位子的角色利益。这是这类实验最常见、也最致命的失效方式。
#
# 做法：把轮换写成条件里的一个键（assign），用 sweep 生成，让 manifest 冻结它。
# 默认置换按 3 个座位定义：
#   L1: 座位1/2/3 = 模型1/2/3   （恒等）
#   L2: 座位1/2/3 = 模型3/1/2
#   L3: 座位1/2/3 = 模型2/3/1
# 三者合起来，每个模型在每个座位上各出现一次 → 模型主效应与角色主效应可分离。
ASSIGN_ROTATIONS = {
    "L1": (0, 1, 2),
    "L2": (2, 0, 1),
    "L3": (1, 2, 0),
    # R1..R3 是同一组置换的别名（核废水场景按 rotation 命名）。
    # 语义必须逐条相等：R2 就是 L2，不要给它们写不同的置换，否则两套命名会悄悄分叉。
    "R1": (0, 1, 2),
    "R2": (2, 0, 1),
    "R3": (1, 2, 0),
}


def rotation_for(assign) -> tuple:
    """把条件里的 assign 解析成置换。未知值直接报错，不静默退回恒等。

    静默退回是最危险的失效模式：实验会跑完、报告会生成、数字看起来都正常，
    只有设计已经悄悄失效了。所以宁可在这里炸。
    """
    key = str(assign or "L1").strip()
    if key not in ASSIGN_ROTATIONS:
        raise ValueError("未知的座位轮换 assign={!r}，可用：{}".format(
            key, "、".join(sorted(ASSIGN_ROTATIONS))))
    return ASSIGN_ROTATIONS[key]


def rotate_seats(seats, assign=None) -> list:
    """按 assign 轮换"座位 → 模型"的对应关系，返回新列表。

    必须返回新列表：seats 是 run_batch 与所有并发线程共享的对象，
    原地 reorder 会让同时开跑的几局互相污染，而且这种 bug 只在并发数 >1 时出现。
    """
    seats = list(seats or [])
    if not seats:
        return []
    # 没指定 assign 就是"不轮换"，而不是"套用默认置换"。
    # 否则一个 2 个座位、根本没要求轮换的批次，会被 3 座位置换判成配置错误。
    if assign is None or str(assign).strip() in ("", "none", "None"):
        return seats
    perm = rotation_for(assign)
    if len(seats) != len(perm):
        raise ValueError(
            "assign={} 的置换按 {} 个座位定义，但传入了 {} 个；"
            "请为该座位数在 ASSIGN_ROTATIONS 里补一条置换（不要靠自动外推，"
            "外推出来的置换未必是真拉丁方）".format(assign or "L1", len(perm), len(seats)))
    if sorted(perm) != list(range(len(perm))):
        raise ValueError("assign={} 的置换不是合法排列：{}".format(assign, perm))
    return [seats[i] for i in perm]


def seat_model_map(seats) -> dict:
    """座位号 → 模型标识；用来把"谁坐了哪个位子"写进结果行与清单。"""
    out = {}
    for idx, seat in enumerate(seats or [], start=1):
        seat = seat or {}
        conf = seat.get("model_conf") or {}
        out[str(idx)] = str(conf.get("model") or seat.get("name") or "")
    return out


def model_specs(seats) -> list:
    """每个座位的模型参数快照（不含密钥）。

    为什么必须冻：temperature、thinking（extra_body）这些值只存在 data/models.json 里，
    那个文件既不进 git、也不进清单。不冻的话，事后无法证明"当时跑的是思考开、温度 1.0"，
    预注册就缺了一块——而"改了参数却看不出来"恰恰是这类实验最难查的一种污染。
    """
    out = []
    for i, seat in enumerate(seats or [], start=1):
        seat = seat or {}
        conf = seat.get("model_conf") or {}
        out.append({
            "seat": i,
            "model": conf.get("model") or seat.get("name") or "",
            "provider": conf.get("provider") or "",
            "protocol": conf.get("protocol") or "",
            "api_base": conf.get("api_base") or "",
            "temperature": conf.get("temperature"),
            "max_tokens": conf.get("max_tokens"),
            "stream": bool(conf.get("stream", False)),
            "extra_body": conf.get("extra_body") or {},
        })
    return out


def validate_assigns(sweep: dict, n_seats: int) -> None:
    """开跑前把 sweep 里所有 assign 值验一遍。

    错在批次启动时暴露一次，而不是在 180 局里各报一次错——后者会产出一份
    "跑完了、有结果"的报告，只有逐行看 state=error 才发现全废。
    """
    for value in (sweep or {}).get("assign") or []:
        perm = rotation_for(value)
        if len(perm) != int(n_seats):
            raise ValueError(
                "assign={} 需要 {} 个座位，实际给了 {} 个座位".format(
                    value, len(perm), n_seats))


def plan_runs(batch_id: str, sweep: dict, seeds) -> list:
    out: list = []
    for cond in build_conditions(sweep):
        cname = condition_name(cond)
        for seed in seeds:
            out.append(RunSpec(run_id="{}-{}-s{}".format(batch_id, cname, seed),
                               condition=cond, seed=int(seed)))
    return out


def _flatten(row: dict, prefix: str, data: dict) -> None:
    for k, v in (data or {}).items():
        if isinstance(v, (dict, list)):
            continue
        row["{}{}".format(prefix, k)] = v


def _execute(plugin, spec, base_config: dict, seats, title: str,
             per_run_timeout: float) -> dict:
    """跑一局，返回一行结果；不碰共享状态，可并发调用。"""
    cfg = dict(base_config or {})
    cfg.update(spec.condition)
    started = time.time()
    row = {"run_id": spec.run_id, "condition": condition_name(spec.condition),
           "seed": spec.seed, "communication": cfg.get("communication", "on"),
           "assign": str(cfg.get("assign", "") or "")}
    try:
        # 每局按 assign 重排座位：同一批次里同一个模型要轮流演不同角色，
        # 否则"模型差异"与"角色差异"分不开。rotate_seats 返回新列表，不动共享的 seats。
        run_seats = rotate_seats(seats, cfg.get("assign"))
        row["seat_models"] = seat_model_map(run_seats)
        label = title or "{} · {} · s{}".format(plugin.name,
                                                condition_name(spec.condition), spec.seed)
        session = runner_mod.create_session(spec.run_id, plugin, cfg, run_seats, label)
        thread = runner_mod.start(session)
        thread.join(timeout=per_run_timeout)
        row["state"] = session.state
        row["winner"] = session.winner or ""
        _flatten(row, "o_", (session.meta or {}).get("outcome") or {})
        _flatten(row, "m_", (session.meta or {}).get("metrics") or {})
        row["messages"] = len(session.messages)
        row["rounds"] = len(session.rounds)
        row["info_visibility"] = cfg.get("info_visibility", "full")
        row["show_reputation"] = cfg.get("show_reputation", "on")
        usages = [((m.meta or {}).get("usage") or {}) for m in session.messages]
        t_in = sum(int(u.get("prompt_tokens") or 0) for u in usages)
        t_out = sum(int(u.get("completion_tokens") or 0) for u in usages)
        row["tokens_in"] = t_in
        row["tokens_out"] = t_out
        row["tokens_total"] = t_in + t_out
        row["usage_estimated"] = any(bool(u.get("estimated")) for u in usages)
        # 额度/限流识别：失败调用里出现 429 的比例
        fails = [str(((m.meta or {}).get("error") or "")) for m in session.messages
                 if (m.meta or {}).get("ok") is False]
        quota = sum(1 for f in fails if "429" in f or "限流" in f)
        row["call_errors"] = len(fails)
        row["quota_errors"] = quota
        if fails and quota >= max(1, int(len(fails) * 0.5)):
            row["state"] = "quota"
        price_in = float(cfg.get("price_in_per_m") or 0.0)
        price_out = float(cfg.get("price_out_per_m") or 0.0)
        if price_in or price_out:
            row["cost_usd"] = round(t_in / 1e6 * price_in + t_out / 1e6 * price_out, 6)
    except Exception as exc:
        row["state"] = "error"
        row["error"] = repr(exc)[:200]
    row["duration_s"] = round(time.time() - started, 1)
    return row


def run_batch(plugin_id: str, *, seats, base_config: dict | None = None,
              sweep: dict | None = None, seeds=(1,), title: str = "",
              out_root=None, resume: bool = True, per_run_timeout: float = 7200.0,
              concurrency: int = 1, on_progress=None) -> dict:
    """执行一个批次；concurrency>1 时多局并行。"""
    plugin = plugin_mod.get_plugin(plugin_id)
    if plugin is None:
        raise ValueError("插件不存在：" + plugin_id)

    seats = list(seats or [])
    # 条件在开跑前就要验：assign 写错要在这一刻炸一次，而不是之后炸满整个批次
    validate_assigns(sweep or {}, len(seats))
    # 把每个 assign 展开成"座位 → 模型"，写进冻结清单
    seat_assign = {str(value): seat_model_map(rotate_seats(seats, value))
                   for value in ((sweep or {}).get("assign") or [])}

    manifest = manifest_mod.freeze(
        plugin=plugin_id, scenario=str((base_config or {}).get("scenario", "")),
        config=base_config or {}, seats=[s.get("model_conf", {}).get("model") or s.get("name")
                                        for s in seats],
        seeds=list(seeds), metrics=["metrics", "outcome"],
        seat_assign=seat_assign,
        model_specs=model_specs(seats),
        notes="sweep=" + str(sweep or {}))
    batch_id = "b" + manifest["digest"][:10]
    out_root = Path(out_root) if out_root else (ROOT / "runs" / "batches")
    out_dir = out_root / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = plan_runs(batch_id, sweep or {}, seeds)
    manifest_path = manifest_mod.save(manifest, str(out_dir))
    manifest["path"] = manifest_path

    index_path = out_dir / "index.csv"
    rows: list = []
    done: set = set()
    if resume and index_path.exists():
        try:
            import csv
            with open(index_path, encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    rows.append(dict(r))
                    if r.get("run_id"):
                        done.add(r["run_id"])
        except Exception:
            rows = []

    pending = [s for s in specs if s.run_id not in done]
    errors = []
    executed = 0
    lock = threading.Lock()

    quota_streak = [0]

    def record(row):
        nonlocal executed
        if row.get("state") == "quota":
            quota_streak[0] += 1
        else:
            quota_streak[0] = 0
        with lock:
            rows.append(row)
            executed += 1
            if row.get("state") == "error":
                errors.append((row["run_id"], row.get("error", "")))
            metrics_mod.write_csv(str(index_path), rows)
        if on_progress:
            on_progress(row, executed, len(pending))

    if concurrency and int(concurrency) > 1 and len(pending) > 1:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = [pool.submit(_execute, plugin, spec, base_config or {}, seats,
                                   title, per_run_timeout) for spec in pending]
            for fut in as_completed(futures):
                record(fut.result())
                if quota_streak[0] >= 2:
                    break
    else:
        for spec in pending:
            record(_execute(plugin, spec, base_config or {}, seats, title, per_run_timeout))
            if quota_streak[0] >= 2:
                break

    if rows:
        metrics_mod.write_csv(str(index_path), rows)

    summary = {"batch_id": batch_id, "index": str(index_path),
               "manifest": manifest_path, "runs": len(specs),
               "executed": executed, "skipped": len(done),
               "concurrency": int(concurrency or 1),
               "aborted": "quota" if quota_streak[0] >= 2 else "",
               "errors": errors, "dir": str(out_dir)}
    try:
        import base.report as report_mod
        md = report_mod.build(rows, manifest, chart_dir=str(out_dir))
        summary["report"] = report_mod.write(str(out_dir / "REPORT.md"), md)
    except Exception as exc:
        summary["report_error"] = repr(exc)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary