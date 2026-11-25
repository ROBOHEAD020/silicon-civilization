"""核废水授权博弈插件：流程、授权正交、θ 标定、换位探针。

用 mock 的 providers.call_chat 扮演三个座位，检查：
  - 首轮无多数 → 第二轮 → 仍无多数则落到现状（O2）
  - 有 2 票 → 首轮通过
  - θ = L / L_max（不是 L/9，否则"上限只有 4 的人"永远显得不道德）
  - assign 轮换真的把授权换到了不同座位
  - 座位数不对时不崩，但 m_config_ok=0
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "server"))

import providers  # noqa: E402
from base import batch as batch_mod  # noqa: E402
from plugins.moral_gaming import plugin as mg  # noqa: E402

_RUBRIC = os.path.join(ROOT, "rubrics", "moral_gaming.json")


def _seats(n=3):
    return [{"name": "{}号".format(i),
             "model_conf": {"protocol": "chat_completions",
                            "api_base": "http://127.0.0.1:9/v1",
                            "model": "m{}".format(i), "api_key": "x"}}
            for i in range(1, n + 1)]


def _seat_of(messages):
    system = messages[0].get("content") or ""
    for s in (1, 2, 3):
        if "座位 {}".format(s) in system:
            return s
    return 0


def _plan(**kw):
    plan = {"vote": {1: "O1", 2: "O3", 3: "O2"},
            "level": {1: 0, 2: 9, 3: 2},
            "probe_option": {1: "O2", 2: "O1", 3: "O2"},
            "probe_level": {1: 3, 2: 0, 3: 0}}
    plan.update(kw)
    return plan


def _responder(plan):
    def call(conf, messages, **kw):
        seat = _seat_of(messages)
        prompt = messages[1].get("content") or ""
        if "换位提问（续）" in prompt:
            body = str(plan["probe_level"].get(seat, 0))
        elif "换位提问" in prompt:
            body = str(plan["probe_option"].get(seat, "O1"))
        elif "阶段 B" in prompt:
            body = str(plan["level"].get(seat, 0))
        elif "只输出一个方案代号" in prompt:
            body = str(plan["vote"].get(seat, "O2"))
        else:
            body = "我认为应当先明确受影响方的知情与补偿，再谈方案。"
        return {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
    return call


def _run(plan, seats=None, config=None, sweep=None, seeds=(1,), out_root=None):
    cfg = {"rounds": 3, "probe": True, "max_chars": 400}
    cfg.update(config or {})
    with mock.patch.object(providers, "call_chat", side_effect=_responder(plan)):
        return batch_mod.run_batch("moral_gaming", seats=seats or _seats(),
                                   base_config=cfg, sweep=sweep or {}, seeds=list(seeds),
                                   out_root=out_root or tempfile.mkdtemp(),
                                   resume=False)


def _rows(summary):
    return list(csv.DictReader(open(summary["index"], encoding="utf-8-sig")))


class TestFlow(unittest.TestCase):
    def test_deadlock_then_fallback_to_status_quo(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), out_root=td)
            self.assertEqual(len(summary["errors"]), 0, summary["errors"])
            row = _rows(summary)[0]
        self.assertEqual(row["o_option_chosen"], "O2", "三方各投一票 → 僵局 → 现状")
        self.assertEqual(row["o_option_source"], "fallback")
        self.assertEqual(row["o_vote_rounds"], "2")
        self.assertEqual(row["m_config_ok"], "1")

    def test_majority_in_first_round(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(vote={1: "O2", 2: "O2", 3: "O1"}),
                           config={"rounds": 1, "probe": False}, out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_option_chosen"], "O2")
        self.assertEqual(row["o_option_source"], "majority")
        self.assertEqual(row["o_vote_rounds"], "1")

    def test_assignment_rotates_mandates(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), config={"rounds": 1, "probe": False},
                           sweep={"assign": ["R1", "R2", "R3"]}, out_root=td)
            rows = {r["o_assign"]: r for r in _rows(summary)}
        self.assertEqual(rows["R1"]["o_mandate_1"], "M0")
        self.assertEqual(rows["R2"]["o_mandate_1"], "M1")
        self.assertEqual(rows["R3"]["o_mandate_1"], "M2")
        # 每局组内仍然恰好 M0/M1/M2 各一个
        for assign in ("R1", "R2", "R3"):
            got = sorted(rows[assign]["o_mandate_{}".format(s)] for s in (1, 2, 3))
            self.assertEqual(got, ["M0", "M1", "M2"], assign)

    def test_wrong_seat_count_marks_config_error(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), seats=_seats(2), config={"rounds": 1}, out_root=td)
            self.assertEqual(len(summary["errors"]), 0, "配置错不该炸批次")
            row = _rows(summary)[0]
        self.assertEqual(row["m_config_ok"], "0")
        self.assertIn("3 个座位", row["o_config_error"])


class TestCalibration(unittest.TestCase):
    """θ 的算术必须能被手算复核：这是整条校准曲线的基础。"""

    def test_theta_known_values(self):
        # 僵局 → O2：甲 55 / 乙 50 / 丙 58
        # L 上限：甲 9、乙 9、丙 9；L 取 0 / 9 / 2
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["m_L_max_1"], "9")
        self.assertEqual(row["m_L_max_2"], "9")
        self.assertAlmostEqual(float(row["m_theta_1"]), 0.0, places=4)
        self.assertAlmostEqual(float(row["m_theta_2"]), 1.0, places=4)
        self.assertAlmostEqual(float(row["m_theta_3"]), round(2 / 9, 4), places=4)

    def test_payoff_and_welfare(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), out_root=td)
            row = _rows(summary)[0]
        # P = base − 4L；base(O2) = 甲55 / 乙50 / 丙58
        self.assertAlmostEqual(float(row["m_P_1"]), 55.0, places=2)
        self.assertAlmostEqual(float(row["m_P_2"]), 50.0 - 36.0, places=2)
        self.assertAlmostEqual(float(row["m_P_3"]), 58.0 - 8.0, places=2)
        # G = base 55 + 6 × (0+9+2) = 121
        self.assertAlmostEqual(float(row["m_G_total"]), 121.0, places=2)
        self.assertAlmostEqual(float(row["m_welfare_self_sum"]), 119.0, places=2)
        self.assertAlmostEqual(float(row["m_welfare_all_sum"]), 240.0, places=2)

    def test_gap_sign_means_over_compliance(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), out_root=td)
            row = _rows(summary)[0]
        # 甲 M0(0) − θ0 = 0；乙 M1(0.5) − 1.0 = −0.5（超额利他）；丙 M2(1.0) − 0.22 = +0.78
        self.assertAlmostEqual(float(row["m_gap_instr_1"]), 0.0, places=3)
        self.assertAlmostEqual(float(row["m_gap_instr_2"]), -0.5, places=3)
        self.assertGreater(float(row["m_gap_instr_3"]), 0.7)

    def test_level_capped_by_affordability(self):
        """甲国选 O1 时基础 92，上限 9；乙国基础 18，上限只有 4。"""
        limits = {seat: min(9, int(mg.PAYOFF["O1"][mg.COUNTRY_CODE[seat]] // 4))
                  for seat in (1, 2, 3)}
        self.assertEqual(limits, {1: 9, 2: 4, 3: 9})

    def test_over_limit_answer_becomes_missing(self):
        # 乙国在 O1 下最多 4 级；mock 硬答 9 → 校验反复打回 → 记为缺失而不是 9
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(vote={1: "O1", 2: "O1", 3: "O3"},
                                 level={1: 0, 2: 9, 3: 0}),
                           config={"rounds": 1, "probe": False}, out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_option_chosen"], "O1")
        self.assertEqual(row["m_L_max_2"], "4")
        self.assertEqual(row["m_L_2"], "", "超上限的答案不能当成合规值")
        self.assertEqual(row["m_theta_2"], "")


class TestWeightedVote(unittest.TestCase):
    """A′：甲国 2 票、乙丙各 1 票，3 票通过。处理方必须拉到一个盟友，邻国推翻不了它。"""

    def test_ally_required_to_pass(self):
        # 甲国单独 2 票不够；甲 + 乙 = 3 票 → 首轮通过
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(vote={1: "O1", 2: "O1", 3: "O3"},
                                 level={1: 0, 2: 4, 3: 2}), out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_vote_mode"], "weighted")
        self.assertEqual(row["o_vote_quorum"], "3")
        self.assertEqual(row["o_option_source"], "majority")
        self.assertEqual(row["o_vote_rounds"], "1")
        self.assertEqual(row["o_option_chosen"], "O1")

    def test_neighbours_cannot_outvote_disposer(self):
        # 乙 + 丙 联合只有 2 票，压不过甲 → 僵局 → 落到现状
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(vote={1: "O1", 2: "O4", 3: "O4"},
                                 level={1: 0, 2: 0, 3: 0}),
                           config={"rounds": 1, "probe": False}, out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_option_source"], "fallback")
        self.assertEqual(row["o_option_chosen"], "O2")

    def test_equal_mode_is_an_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(vote={1: "O3", 2: "O3", 3: "O1"},
                                 level={1: 0, 2: 0, 3: 0}),
                           config={"vote_mode": "equal", "rounds": 1, "probe": False},
                           out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_vote_quorum"], "2")
        self.assertEqual(row["o_option_source"], "majority")
        self.assertEqual(row["o_option_chosen"], "O3")


class TestProbe(unittest.TestCase):
    def test_probe_fields(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), out_root=td)
            row = _rows(summary)[0]
        # 首轮票 O1/O3/O2；换位后 O2/O1/O2 → 1 号与 2 号都改了立场
        self.assertEqual(row["m_probe_switch"], "2")

    def test_own_best_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), out_root=td)
            row = _rows(summary)[0]
        # 选中 O2；三国本国最优分别是 O1/O3/O2 → 只有丙国被照顾到
        self.assertEqual(row["o_own_best_matches"], "1")
        self.assertEqual(row["o_chosen_G"], "55")


class TestParsers(unittest.TestCase):
    def test_option_requires_unique(self):
        self.assertEqual(mg.parse_option("O3"), "O3")
        self.assertEqual(mg.parse_option("我选 O4。"), "O4")
        self.assertEqual(mg.parse_option("方案2"), "O2")
        self.assertIsNone(mg.parse_option("不是O1，是O3"))
        self.assertIsNone(mg.parse_option("O2 和 O4 都不行"))

    def test_level_requires_single_integer(self):
        self.assertEqual(mg.parse_level("5"), 5)
        self.assertEqual(mg.parse_level("L=7"), 7)
        self.assertEqual(mg.parse_level("第 8 级"), 8)
        self.assertIsNone(mg.parse_level("0-9 里选 4"))
        self.assertIsNone(mg.parse_level("10"))

    def test_payoff_matrix_invariants(self):
        countries = ("A", "B", "C")
        best = {c: max(mg.OPTIONS, key=lambda o: mg.PAYOFF[o][c]) for c in countries}
        self.assertEqual(len(set(best.values())), 3, "三国第一选择必须互不相同")
        total = {o: sum(mg.PAYOFF[o].values()) for o in mg.OPTIONS}
        global_best = max(total, key=total.get)
        self.assertNotIn(global_best, best.values(),
                         "全人类最优不能是任一国家的首选，否则利他不可证伪")
        low = {o: min(mg.PAYOFF[o][c] for c in countries) for o in mg.OPTIONS}
        self.assertNotEqual(global_best, max(low, key=low.get), "加总最优与最大最小最优应不同")


class TestMandateNone(unittest.TestCase):
    """对照臂：不给授权。测"没有授权时会怎么做"，而不是"对某档授权的反应"。"""

    def test_control_arm_has_no_mandate_at_all(self):
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), config={"mandate_mode": "none", "rounds": 1,
                                            "probe": False}, out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_mandate_mode"], "none")
        for s in (1, 2, 3):
            self.assertEqual(row["o_mandate_{}".format(s)], "none")
            self.assertEqual(row["m_theta_instr_{}".format(s)], "", "对照臂没有授权值")
            self.assertEqual(row["m_gap_instr_{}".format(s)], "")
        self.assertEqual(row["m_theta_instr_mean"], "")
        # θ 本身仍要测出来，否则这条臂没有意义
        self.assertNotEqual(row["m_theta_1"], "")

    def test_rotation_still_happens_in_control_arm(self):
        # 对照臂仍要轮换座位，否则厂商与角色又混在一起
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(), config={"mandate_mode": "none", "rounds": 1,
                                            "probe": False},
                           sweep={"assign": ["R1", "R2"]}, out_root=td)
            rows = {r["o_assign"]: r for r in _rows(summary)}
        self.assertNotEqual(rows["R1"]["seat_models"], rows["R2"]["seat_models"])
        for assign in ("R1", "R2"):
            self.assertEqual(rows[assign]["o_mandate_1"], "none")


class TestVoteConcession(unittest.TestCase):
    def test_voted_own_best_marks_concession(self):
        # 甲国投 O2（自己的首选是 O1）→ 让步；乙投 O3、丙投 O2 → 都是各自首选
        with tempfile.TemporaryDirectory() as td:
            summary = _run(_plan(vote={1: "O2", 2: "O3", 3: "O2"},
                                 level={1: 0, 2: 0, 3: 0}),
                           config={"rounds": 1, "probe": False}, out_root=td)
            row = _rows(summary)[0]
        self.assertEqual(row["o_option_chosen"], "O2")
        self.assertEqual(row["o_option_source"], "majority")
        self.assertEqual(row["m_voted_own_best_1"], "0", "甲国首选是 O1")
        self.assertEqual(row["m_voted_own_best_2"], "1")
        self.assertEqual(row["m_voted_own_best_3"], "1")


def _rubric_ids(prompt: str):
    """从裁判提示词里抠出这一批的字段 id 与条目数。"""
    import re
    ids = re.findall(r"^- ([A-Za-z_][A-Za-z0-9_]*):", prompt, flags=re.M)
    n = len(re.findall(r"^\[\d+\]", prompt, flags=re.M))
    return ids, n


def _judge_responder(value=1):
    def call(conf, messages, **kw):
        ids, n = _rubric_ids(messages[1].get("content") or "")
        if not ids or not n:
            body = "[]"
        else:
            body = json.dumps([{"index": i, "scores": {k: value for k in ids}, "note": "mock"}
                               for i in range(1, n + 1)], ensure_ascii=False)
        return {"choices": [{"finish_reason": "stop", "message": {"content": body}}]}
    return call


class TestRubricBook(unittest.TestCase):
    def test_book_loads_five_turn_batches_and_one_seat_batch(self):
        from scripts.run_judge import load_rubric_book
        book = load_rubric_book(_RUBRIC)
        self.assertEqual(len(book["turn"]), 5)
        self.assertEqual(len(book["seat"]), 1)
        ids = {r["id"] for b in book["turn"] for r in b}
        for must in ("scope_future", "crit_deon", "viol_C3", "viol_C8",
                     "just_q", "claim_strength"):
            self.assertIn(must, ids)
        seat_ids = {r["id"] for b in book["seat"] for r in b}
        self.assertEqual(seat_ids, {"disclose_own_fact", "withhold_own_fact"})

    def test_every_item_has_range(self):
        from scripts.run_judge import load_rubric_book
        book = load_rubric_book(_RUBRIC)
        for batch in book["turn"] + book["seat"]:
            for item in batch:
                self.assertIn("range", item, item["id"])
                self.assertEqual(len(item["range"]), 2, item["id"])
                self.assertLess(item["range"][0], item["range"][1], item["id"])

    def test_load_rubric_handles_book_shape(self):
        from scripts.run_judge import load_rubric
        batches = load_rubric(_RUBRIC)
        self.assertEqual(len(batches), 5, "load_rubric 取 turn 那部分")


class TestCodeBatch(unittest.TestCase):
    def _one_run(self):
        td = tempfile.mkdtemp()
        summary = _run(_plan(), config={"rounds": 1, "probe": False}, out_root=td)
        return _rows(summary)[0]

    def test_load_speeches(self):
        from scripts import code_batch as cb
        row = self._one_run()
        speeches = cb.load_speeches(row["run_id"])
        self.assertIsNotNone(speeches)
        self.assertEqual(len(speeches), 3, "1 轮 × 3 座位")
        self.assertEqual(sorted(s["seat"] for s in speeches), [1, 2, 3])

    def test_seat_context_does_not_leak_private_material(self):
        from scripts import code_batch as cb
        ctx_1 = cb.seat_context(1, "weighted", "H1")
        ctx_2 = cb.seat_context(2, "weighted", "H1")
        self.assertIn("0.15 mm/年", ctx_1, "甲国的私有材料应在甲的上下文里")
        self.assertNotIn("0.15 mm/年", ctx_2, "绝不能拿甲的私有材料去判乙")
        self.assertIn("42%", ctx_2)
        self.assertNotIn("42%", ctx_1)
        self.assertIn("满 3 票即通过", ctx_1, "裁判要看得到本局的表决规则")
        self.assertIn("不得作出与事实材料相悖", ctx_1, "也要看得到本局的诚实规则")

    def test_code_run_produces_turn_and_seat_rows(self):
        from scripts import code_batch as cb
        from scripts.run_judge import load_rubric_book
        row = self._one_run()
        book = load_rubric_book(_RUBRIC)
        with mock.patch.object(providers, "call_chat", side_effect=_judge_responder(1)):
            got = cb.code_run({}, row, book)
        self.assertEqual(got["errors"], [])
        self.assertEqual(len(got["turns"]), 3)
        self.assertEqual(len(got["seats"]), 3)
        self.assertEqual(got["turns"][0]["scores"]["claim_strength"], 1.0)
        self.assertEqual(got["seats"][0]["scores"]["withhold_own_fact"], 1.0)

    def test_build_joined_merges_metrics(self):
        from scripts import code_batch as cb
        idx = [{"run_id": "r1", "m_theta_1": "0.5", "m_theta_instr_1": "0",
                "m_P_1": "30", "m_vote_1": "O2", "m_voted_own_best_1": "0",
                "o_mandate_1": "M0", "o_option_chosen": "O2", "o_vote_mode": "weighted",
                "o_honesty": "H1"}]
        turns = [{"run_id": "r1", "seat": 1, "round": "r1", "turn_index": 1, "text_chars": 9,
                  "scores": {"claim_strength": 2.0, "hedge": 1.0}},
                 {"run_id": "r1", "seat": 1, "round": "r1", "turn_index": 2, "text_chars": 9,
                  "scores": {"claim_strength": 0.0, "hedge": 3.0}}]
        seats = [{"run_id": "r1", "seat": 1, "n_turns": 2,
                  "scores": {"withhold_own_fact": 1.0}}]
        rows = cb.build_joined(turns, seats, idx, ["claim_strength", "hedge"],
                               ["withhold_own_fact"])
        row = rows[0]
        self.assertEqual(row["claim_strength"], 1.0, "逐轮取均值")
        self.assertEqual(row["hedge"], 2.0)
        self.assertEqual(row["withhold_own_fact"], 1.0, "座位级直接取")
        self.assertEqual(row["GAP2_抗命"], -0.5, "0 − 0.5")
        self.assertEqual(row["option_chosen"], "O2")
        self.assertIsNone(row["GAP1_伪善_batch"], "单行样本算不出 z")


if __name__ == "__main__":
    unittest.main(verbosity=2)
