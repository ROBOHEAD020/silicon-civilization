"""投票决原语（场景无关）。

最小模型：一轮一票，票投给若干候选之一；达到法定票数即通过；
未达到则进入下一轮；轮次用尽则落到默认结局（通常是"现状延续"）。

支持加权：一张票可以值 2 分（例如"处理方 2 票、邻国各 1 票"）。
加权不是花活，它决定谁必须拉盟友：等权下两家联合就能推翻处理方，
加权下不能——这直接改变了"劝阻"有没有分量。

为什么单独抽出来：多数决 + 第二轮 + 默认项这套规则本身是场景的实质内容——
它决定"坚持立场"是否免费、僵局由谁承担代价。埋在某个插件里会让
跨场景的指标口径无法比较，也让"平票怎么处理"变成没人检查的隐含假设。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ballot:
    """一轮票型。"""
    round_no: int
    votes: dict = field(default_factory=dict)      # seat -> option

    def to_dict(self) -> dict:
        return {"round": self.round_no,
                "votes": {str(k): v for k, v in sorted(self.votes.items())}}


def tally(votes: dict) -> dict:
    """人头计数：{选项: 票数}，按票数降序、选项名升序。"""
    counts: dict = {}
    for opt in (votes or {}).values():
        counts[opt] = counts.get(opt, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def weighted_tally(votes: dict, weights: dict | None = None) -> dict:
    """加权票值：{选项: 加权后的票数}。没给权重就等同于 tally。"""
    weights = weights or {}
    out: dict = {}
    for seat, opt in (votes or {}).items():
        try:
            w = float(weights.get(int(seat), 1.0))
        except (TypeError, ValueError):
            w = 1.0
        out[opt] = out.get(opt, 0.0) + w
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def majority(votes: dict, quorum, weights: dict | None = None) -> str | None:
    """唯一达到法定票数的选项。

    给了 weights 就按加权票值计（例如"甲国 2 票、乙国 1 票"），没给就一人一票。
    平票（两个及以上都达标）与无人达标都返回 None——这两种情况在
    "该不该进第二轮"上是同一件事，合并处理可以避免给场景留歧义。
    """
    counts = weighted_tally(votes, weights) if weights else tally(votes)
    gate = float(quorum)
    reached = [opt for opt, n in counts.items() if n >= gate - 1e-9]
    return reached[0] if len(reached) == 1 else None


class QuorumVote:
    """一组有法定票数与默认结局的投票。

    典型用法（三轮表决、加权 3 票通过、僵局落到 O2）：

        vote = QuorumVote(["O1", "O2", "O3", "O4"], quorum=3, fallback="O2",
                          max_rounds=2, weights={1: 2, 2: 1, 3: 1})
        while vote.outcome()["source"] == "pending":
            vote.cast_round({seat: ask_seat(seat) for seat in seats})
        result = vote.outcome()     # option / source / decided_round

    fallback 为 None 表示"僵局就是没有决定"，由场景自行处理。
    """

    def __init__(self, options, quorum: int = 2, fallback: str | None = None,
                 max_rounds: int = 2, weights: dict | None = None):
        self.options = [str(o) for o in (options or [])]
        if not self.options:
            raise ValueError("QuorumVote 需要至少一个候选")
        self.quorum = int(quorum)
        if self.quorum < 1:
            raise ValueError("quorum 必须 >= 1")
        self.fallback = None if fallback is None else str(fallback)
        if self.fallback is not None and self.fallback not in self.options:
            raise ValueError("fallback={!r} 不在候选中".format(self.fallback))
        self.max_rounds = max(1, int(max_rounds))
        clean = {}
        for seat, w in (weights or {}).items():
            try:
                clean[int(seat)] = float(w)
            except (TypeError, ValueError):
                continue
        self.weights = clean
        self.ballots: list = []
        if self.weights:
            total = sum(self.weights.values())
            if self.quorum > total + 1e-9:
                raise ValueError(
                    "法定票数 {} 超过全部票值 {}，任何投票都无法通过".format(self.quorum, total))

    # ---------- 记录 ----------
    def cast_round(self, votes: dict) -> Ballot:
        """记一轮票并冻结。非法选项直接报错——投票静默作废会让结果无从解释。"""
        clean: dict = {}
        for seat, opt in (votes or {}).items():
            if str(opt) not in self.options:
                raise ValueError("座位 {} 投了非法选项 {!r}".format(seat, opt))
            clean[int(seat)] = str(opt)
        if not clean:
            raise ValueError("空票：一轮里没有任何合法投票")
        ballot = Ballot(len(self.ballots) + 1, clean)
        self.ballots.append(ballot)
        return ballot

    # ---------- 查询 ----------
    @property
    def rounds(self) -> int:
        return len(self.ballots)

    def last_tally(self) -> dict:
        """最近一轮的加权票值。"""
        if not self.ballots:
            return {}
        return weighted_tally(self.ballots[-1].votes, self.weights)

    def winner(self) -> str | None:
        """按已记录的轮次，胜者是否已确定。"""
        for ballot in self.ballots:
            win = majority(ballot.votes, self.quorum, self.weights)
            if win is not None:
                return win
        return None

    def _result(self, option, source, decided_round, ballot) -> dict:
        votes = (ballot.votes if ballot else {})
        return {"option": option, "source": source, "decided_round": decided_round,
                "quorum": self.quorum, "fallback": self.fallback,
                "weights": {str(k): v for k, v in sorted(self.weights.items())},
                # tally 是加权票值（决定成败的那个数）；
                # seat_tally 是人头数（写报告时说"两人支持"用的那个数）。两个都要留。
                "tally": weighted_tally(votes, self.weights),
                "seat_tally": tally(votes),
                "ballots": [b.to_dict() for b in self.ballots]}

    def outcome(self) -> dict:
        """最终结论。

        source 三态：
          majority —— 某轮唯一达标，option 为该选项
          fallback —— 轮次用尽仍无多数，option 为默认项（可能为 None）
          pending  —— 还没投够轮次，option 为 None
        """
        for i, ballot in enumerate(self.ballots, start=1):
            win = majority(ballot.votes, self.quorum, self.weights)
            if win is not None:
                return self._result(win, "majority", i, ballot)
        if self.rounds >= self.max_rounds:
            return self._result(self.fallback, "fallback", None,
                                self.ballots[-1] if self.ballots else None)
        return self._result(None, "pending", None,
                            self.ballots[-1] if self.ballots else None)

    def stats(self) -> dict:
        """写进结果行的扁平指标。"""
        out = self.outcome()
        tally_now = out.get("tally") or {}
        return {
            "vote_rounds": self.rounds,
            "vote_option": out["option"] or "",
            "vote_source": out["source"],
            "vote_stalemate": 1 if out["source"] == "fallback" else 0,
            "vote_fragmentation": len(tally_now),
            "vote_points_top": max(tally_now.values()) if tally_now else 0.0,
        }
