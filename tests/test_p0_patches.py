"""P0/P1 补丁的回归测试：座位轮换、裁判截断与分批、投票决。

覆盖三处会"静默失准"的地方：
  1. 座位↔模型不轮换 → 模型效应与角色效应混淆（设计失效但结果看着正常）；
  2. 裁判 400 字符硬截断 → 论证结构被砍，编码数据从源头失真；
  3. 多数决没有原语 → 平票/僵局规则散落在插件里，口径无法跨场景比较。
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
import base.judge as judge_mod  # noqa: E402
import base.manifest as manifest_mod  # noqa: E402
import base.vote as vote_mod  # noqa: E402
from base import batch as batch_mod  # noqa: E402


def _seats(models=("m1", "m2", "m3")):
    return [{"name": "{}号".format(i),
             "model_conf": {"protocol": "chat_completions",
                            "api_base": "http://127.0.0.1:9/v1",
                            "model": m, "api_key": "x"}}
            for i, m in enumerate(models, start=1)]


def _conf():
    return {"protocol": "chat_completions", "api_base": "http://127.0.0.1:9/v1",
            "model": "judge-x", "api_key": "x"}


def fake_call(conf, messages, **kw):
    return {"choices": [{"finish_reason": "stop",
                         "message": {"content": "我认为应当先公开监测数据。"}}]}


# ---------------------------------------------------------------- 座位轮换
class TestSeatRotation(unittest.TestCase):
    def test_identity(self):
        seats = _seats()
        out = batch_mod.rotate_seats(seats, "L1")
        self.assertEqual([s["model_conf"]["model"] for s in out], ["m1", "m2", "m3"])

    def test_rotations_are_orthogonal(self):
        """三个置换合起来，每个模型在每个座位上恰好出现一次。"""
        seats = _seats()
        seen = {}
        for assign in ("L1", "L2", "L3"):
            mapping = batch_mod.seat_model_map(batch_mod.rotate_seats(seats, assign))
            self.assertEqual(sorted(mapping.values()), ["m1", "m2", "m3"])
            for pos, model in mapping.items():
                seen.setdefault(model, set()).add(pos)
        for model, positions in seen.items():
            self.assertEqual(positions, {"1", "2", "3"},
                             "{} 没有在每个座位上各坐一次：{}".format(model, positions))

    def test_does_not_mutate_input(self):
        seats = _seats()
        before = [s["model_conf"]["model"] for s in seats]
        batch_mod.rotate_seats(seats, "L2")
        self.assertEqual([s["model_conf"]["model"] for s in seats], before,
                         "rotate_seats 不能原地改共享的 seats")

    def test_unknown_assign_raises(self):
        with self.assertRaises(ValueError):
            batch_mod.rotate_seats(_seats(), "L9")

    def test_wrong_seat_count_raises(self):
        with self.assertRaises(ValueError):
            batch_mod.rotate_seats(_seats(("a", "b")), "L1")

    def test_seat_model_map(self):
        self.assertEqual(batch_mod.seat_model_map(_seats()), {"1": "m1", "2": "m2", "3": "m3"})

    def test_validate_assigns(self):
        batch_mod.validate_assigns({"assign": ["L1", "L2"]}, 3)
        with self.assertRaises(ValueError):
            batch_mod.validate_assigns({"assign": ["L1"]}, 2)

    def test_r_aliases_match_l(self):
        for k in ("1", "2", "3"):
            self.assertEqual(batch_mod.rotation_for("R" + k),
                             batch_mod.rotation_for("L" + k),
                             "R{} 与 L{} 必须是同一个置换".format(k, k))

    def test_no_assign_means_no_rotation(self):
        out = batch_mod.rotate_seats(_seats(), None)
        self.assertEqual([s["model_conf"]["model"] for s in out], ["m1", "m2", "m3"],
                         "没给 assign 就是不动，不能套用默认置换")


# ---------------------------------------------------------------- 批次端到端
class TestBatchRotatesSeats(unittest.TestCase):
    def test_each_run_records_its_own_mapping(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(providers, "call_chat", side_effect=fake_call):
            summary = batch_mod.run_batch(
                "discussion", seats=_seats(),
                base_config={"rounds": 1, "topic": "测试"},
                sweep={"assign": ["L1", "L2", "L3"]}, seeds=[1],
                out_root=td, resume=False)
            self.assertEqual(summary["executed"], 3)
            self.assertEqual(len(summary["errors"]), 0)
            rows = list(csv.DictReader(open(summary["index"], encoding="utf-8-sig")))
            got = {r["assign"]: json.loads(r["seat_models"]) for r in rows}
            self.assertEqual(got["L1"], {"1": "m1", "2": "m2", "3": "m3"})
            self.assertEqual(got["L2"], {"1": "m3", "2": "m1", "3": "m2"})
            self.assertEqual(got["L3"], {"1": "m2", "2": "m3", "3": "m1"})

    def test_manifest_freezes_the_mapping(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(providers, "call_chat", side_effect=fake_call):
            summary = batch_mod.run_batch(
                "discussion", seats=_seats(), base_config={"rounds": 1},
                sweep={"assign": ["L1", "L2"]}, seeds=[1],
                out_root=td, resume=False)
            man = json.load(open(summary["manifest"], encoding="utf-8"))
            self.assertEqual(man["seat_assign"]["L2"], {"1": "m3", "2": "m1", "3": "m2"})
            self.assertTrue(manifest_mod.verify(man), "加了字段后清单仍应自校验通过")

    def test_manifest_freezes_model_params(self):
        """温度与 thinking 必须进清单：它们只存在 models.json 里，不在 git 里。"""
        seats = _seats()
        seats[0]["model_conf"]["temperature"] = 0.6
        seats[0]["model_conf"]["extra_body"] = {"thinking": {"type": "enabled"}}
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(providers, "call_chat", side_effect=fake_call):
            summary = batch_mod.run_batch(
                "discussion", seats=seats, base_config={"rounds": 1},
                seeds=[1], out_root=td, resume=False)
            man = json.load(open(summary["manifest"], encoding="utf-8"))
        spec = man["model_specs"][0]
        self.assertEqual(spec["seat"], 1)
        self.assertEqual(spec["model"], "m1")
        self.assertEqual(spec["temperature"], 0.6)
        self.assertEqual(spec["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("api_key", spec, "清单里绝不能出现密钥")
        self.assertTrue(manifest_mod.verify(man))

    def test_bad_assign_fails_before_running(self):
        with tempfile.TemporaryDirectory() as td, \
                mock.patch.object(providers, "call_chat", side_effect=fake_call):
            with self.assertRaises(ValueError):
                batch_mod.run_batch("discussion", seats=_seats(),
                                    base_config={"rounds": 1},
                                    sweep={"assign": ["L1", "L9"]}, seeds=[1],
                                    out_root=td, resume=False)

    def test_manifest_digest_covers_seat_assign(self):
        base = dict(plugin="p", scenario="", config={}, seats=["m1", "m2", "m3"],
                    seeds=[1], metrics=["metrics"])
        a = manifest_mod.freeze(seat_assign={"L1": {"1": "m1"}}, **base)
        b = manifest_mod.freeze(seat_assign={"L1": {"1": "m3"}}, **base)
        self.assertNotEqual(a["digest"], b["digest"], "换座位映射应当改变条件哈希")
        self.assertIn("seat_assign", manifest_mod.compare(a, b))


# ---------------------------------------------------------------- 裁判
class TestJudgeFixes(unittest.TestCase):
    def test_no_truncation_by_default(self):
        long_text = "论" * 2000
        msgs = judge_mod.build_messages([{"seat": 1, "text": long_text}])
        self.assertIn(long_text, msgs[1]["content"],
                      "默认不应截断：被砍掉的往往正是论证结构")

    def test_explicit_truncation_still_works(self):
        long_text = "论" * 2000
        msgs = judge_mod.build_messages([{"seat": 1, "text": long_text}], max_chars=100)
        self.assertNotIn(long_text, msgs[1]["content"])
        self.assertIn("论" * 100, msgs[1]["content"])

    def test_binary_rubric_example_is_zero(self):
        rubric = [{"id": "scope_local", "q": "是否提及", "range": [0, 1]}]
        text = judge_mod.build_messages([{"seat": 1, "text": "x"}], rubric=rubric)[1]["content"]
        self.assertIn('"scope_local": 0', text, "二元条目的示例不能给 3，否则裁判会往中间凑")
        self.assertIn("取值 0–1", text)

    def test_level_rubric_keeps_midpoint(self):
        rubric = [{"id": "just_q", "q": "质量", "range": [0, 3]}]
        text = judge_mod.build_messages([{"seat": 1, "text": "x"}], rubric=rubric)[1]["content"]
        self.assertIn('"just_q": 3', text)

    def test_judge_passes_max_chars_through(self):
        seen = {}

        def capture(conf, messages, **kw):
            seen["content"] = messages[1]["content"]
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": "[]"}}]}

        long_text = "论" * 500
        with mock.patch.object(providers, "call_chat", side_effect=capture):
            judge_mod.judge(_conf(), [{"seat": 1, "text": long_text}], max_chars=50)
        self.assertNotIn(long_text, seen["content"])

    def test_judge_batches_merges_fields(self):
        def fake(conf, messages, **kw):
            text = messages[1]["content"]
            ids = ["scope_local", "scope_future"] if "scope_local" in text \
                else ["viol_C1", "just_q"]
            payload = [{"index": 1, "scores": {i: 1 for i in ids}, "note": "第一条"},
                       {"index": 2, "scores": {i: 0 for i in ids}, "note": "第二条"}]
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": json.dumps(payload, ensure_ascii=False)}}]}

        rubrics = [
            [{"id": "scope_local", "q": "范围", "range": [0, 1]},
             {"id": "scope_future", "q": "后代", "range": [0, 1]}],
            [{"id": "viol_C1", "q": "不可逆", "range": [0, 1]},
             {"id": "just_q", "q": "质量", "range": [0, 3]}],
        ]
        items = [{"seat": 1, "text": "甲"}, {"seat": 2, "text": "乙"}]
        with mock.patch.object(providers, "call_chat", side_effect=fake):
            result = judge_mod.judge_batches(_conf(), items, rubrics)
        self.assertEqual(len(result["scores"]), 2)
        self.assertEqual(set(result["scores"][0]["scores"]),
                         {"scope_local", "scope_future", "viol_C1", "just_q"})
        self.assertEqual(result["aggregate"]["n"], 2)
        self.assertEqual(result["failed_batches"], [])
        self.assertIn("batch-1", result["scores"][0]["notes"])

    def test_judge_batches_survives_one_failed_batch(self):
        calls = {"n": 0}

        def flaky(conf, messages, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("网关 502")
            payload = [{"index": 1, "scores": {"scope_local": 1}}]
            return {"choices": [{"finish_reason": "stop",
                                 "message": {"content": json.dumps(payload)}}]}

        rubrics = [[{"id": "scope_local", "q": "范围", "range": [0, 1]}],
                   [{"id": "viol_C1", "q": "不可逆", "range": [0, 1]}]]
        with mock.patch.object(providers, "call_chat", side_effect=flaky):
            result = judge_mod.judge_batches(_conf(), [{"seat": 1, "text": "甲"}], rubrics)
        self.assertEqual(result["failed_batches"], [2], "失败批次要能被指出来")
        self.assertIn("scope_local", result["scores"][0]["scores"])


# ---------------------------------------------------------------- 投票
class TestVote(unittest.TestCase):
    def test_tally_order(self):
        self.assertEqual(list(vote_mod.tally({1: "O1", 2: "O1", 3: "O2"})),
                         ["O1", "O2"])

    def test_majority_requires_unique(self):
        self.assertEqual(vote_mod.majority({1: "O1", 2: "O1", 3: "O2"}, 2), "O1")
        # 4 人局里 2:2 时两个选项都达标，算平票、不算通过（3 人局不可能平票）
        self.assertIsNone(vote_mod.majority({1: "O1", 2: "O1", 3: "O2", 4: "O2"}, 2),
                          "两人同时达标算平票，不算通过")

    def test_majority_in_first_round(self):
        vote = vote_mod.QuorumVote(["O1", "O2", "O3"], quorum=2, fallback="O2")
        vote.cast_round({1: "O1", 2: "O1", 3: "O3"})
        out = vote.outcome()
        self.assertEqual((out["option"], out["source"], out["decided_round"]),
                         ("O1", "majority", 1))

    def test_deadlock_goes_to_second_round_then_fallback(self):
        vote = vote_mod.QuorumVote(["O1", "O2", "O3"], quorum=2,
                                   fallback="O2", max_rounds=2)
        vote.cast_round({1: "O1", 2: "O2", 3: "O3"})
        self.assertEqual(vote.outcome()["source"], "pending")
        vote.cast_round({1: "O1", 2: "O2", 3: "O3"})
        out = vote.outcome()
        self.assertEqual(out["source"], "fallback")
        self.assertEqual(out["option"], "O2")
        self.assertEqual(vote.stats()["vote_stalemate"], 1)

    def test_fallback_none_means_no_decision(self):
        vote = vote_mod.QuorumVote(["O1", "O2"], quorum=2, fallback=None, max_rounds=1)
        vote.cast_round({1: "O1", 2: "O2"})
        self.assertIsNone(vote.outcome()["option"])

    def test_illegal_option_rejected(self):
        vote = vote_mod.QuorumVote(["O1", "O2"], quorum=2)
        with self.assertRaises(ValueError):
            vote.cast_round({1: "O9"})
        with self.assertRaises(ValueError):
            vote.cast_round({})

    def test_fallback_must_be_a_candidate(self):
        with self.assertRaises(ValueError):
            vote_mod.QuorumVote(["O1"], quorum=1, fallback="O9")

    def test_stats_keys(self):
        vote = vote_mod.QuorumVote(["O1", "O2"], quorum=2, fallback="O2")
        vote.cast_round({1: "O1", 2: "O1", 3: "O2"})
        stats = vote.stats()
        self.assertEqual(stats["vote_option"], "O1")
        self.assertEqual(stats["vote_rounds"], 1)
        self.assertEqual(stats["vote_fragmentation"], 2)

    # ---- 加权表决：处理方 2 票、邻国各 1 票 ----
    def test_weighted_tally_and_majority(self):
        w = {1: 2, 2: 1, 3: 1}
        self.assertEqual(vote_mod.weighted_tally({1: "O1", 2: "O1", 3: "O2"}, w),
                         {"O1": 3.0, "O2": 1.0})
        # 甲国单独 2 票，够不到 3 → 不算通过
        self.assertIsNone(vote_mod.majority({1: "O1", 2: "O2", 3: "O3"}, 3, w))
        # 甲 + 乙 = 3 → 通过
        self.assertEqual(vote_mod.majority({1: "O1", 2: "O1", 3: "O3"}, 3, w), "O1")
        # 乙 + 丙 = 2，压不过甲 → 平票、仍不通过（关键：邻国推翻不了处理方）
        self.assertIsNone(vote_mod.majority({1: "O1", 2: "O2", 3: "O2"}, 3, w))

    def test_weighted_quorum_vote_deadlock_and_fallback(self):
        w = {1: 2, 2: 1, 3: 1}
        vote = vote_mod.QuorumVote(["O1", "O2"], quorum=3, fallback="O2",
                                   max_rounds=2, weights=w)
        vote.cast_round({1: "O1", 2: "O2", 3: "O2"})
        out = vote.outcome()
        self.assertEqual(out["source"], "pending")
        self.assertEqual(out["tally"], {"O1": 2.0, "O2": 2.0}, "加权票值")
        self.assertEqual(out["seat_tally"], {"O1": 1, "O2": 2}, "人头票：两个数都要留")
        vote.cast_round({1: "O1", 2: "O2", 3: "O2"})
        self.assertEqual(vote.outcome()["source"], "fallback")
        self.assertEqual(vote.outcome()["option"], "O2")

    def test_weighted_quorum_above_total_rejected(self):
        with self.assertRaises(ValueError):
            vote_mod.QuorumVote(["O1"], quorum=5, weights={1: 2, 2: 1, 3: 1})


# ---------------------------------------------------------------- 评分表读取
class TestRubricFile(unittest.TestCase):
    def _write(self, td, payload):
        path = os.path.join(td, "rubric.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return path

    def test_single_rubric_is_wrapped_into_one_batch(self):
        from scripts.run_judge import load_rubric
        rubric = [{"id": "cooperation", "q": "克制", "range": [1, 5]}]
        with tempfile.TemporaryDirectory() as td:
            batches = load_rubric(self._write(td, rubric))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0]["id"], "cooperation")

    def test_batched_rubric_kept_as_batches(self):
        from scripts.run_judge import load_rubric
        payload = [
            [{"id": "scope_local", "q": "范围", "range": [0, 1]}],
            [{"id": "viol_C1", "q": "不可逆", "range": [0, 1]},
             {"id": "just_q", "q": "质量", "range": [0, 3]}],
        ]
        with tempfile.TemporaryDirectory() as td:
            batches = load_rubric(self._write(td, payload))
        self.assertEqual([len(b) for b in batches], [1, 2])

    def test_mixed_shape_rejected(self):
        from scripts.run_judge import load_rubric
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, [{"id": "a", "q": ""}, [{"id": "b", "q": ""}]])
            with self.assertRaises(SystemExit):
                load_rubric(path)

    def test_empty_rejected(self):
        from scripts.run_judge import load_rubric
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit):
                load_rubric(self._write(td, []))


if __name__ == "__main__":
    unittest.main(verbosity=2)