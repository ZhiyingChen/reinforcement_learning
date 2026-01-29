# -*- coding: utf-8 -*-
from __future__ import annotations

from .domain_object import Action, Transition, Reward, Coord
from typing import Dict, Tuple, List, Optional, Iterable
import numpy as np


class GridWorld:
    """
    一个可配置的网格世界（MDP）环境。

    关键能力：
      - π(a|s): 每个 state 的动作分布（可选）
      - P(s'|s,a): 转移概率
      - P(r|s,a,s'): 奖励分布

    约定：
      - forbidden/target 均属于状态空间的一部分。
      - 进入 forbidden 视为终止（默认）；到达 target 也为终止。
      - 命中边界时，保持原地（可修改）。
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        forbidden: Optional[Iterable[Coord]] = None,
        target: Optional[Coord] = None,
        start: Coord = (0, 0),
        absorbing_forbidden: bool = False,
        stay_on_wall: bool = True,
        seed: Optional[int] = None,
    ):
        assert height > 0 and width > 0
        self.h, self.w = height, width
        self.forbidden: set[Coord] = set(forbidden or [])
        self.target: Optional[Coord] = target
        self.start: Coord = start
        self.absorbing_forbidden = absorbing_forbidden
        self.stay_on_wall = stay_on_wall

        # 构造状态编码
        self.id2s: List[Coord] = [(row, col) for row in range(self.h) for col in range(self.w)]
        self.s2id: Dict[Coord, int] = {state: sid for sid, state in enumerate(self.id2s)}
        assert self._in_bounds(start), "起点不在网格内"

        # 当前状态
        self.s: Coord = start

        # 随机数
        self.rng = np.random.default_rng(seed)

        # π(a|s): state -> {action: prob}
        self.policy: Dict[int, Dict[Action, float]] = {}

        # P(s'|s,a): (sid, action) -> {sid': prob}
        self.transitions: Dict[Tuple[int, Action], Dict[int, float]] = {}

        # P(r|s,a,s'): (sid, action, sid') -> {reward: prob}
        self.rewards: Dict[Tuple[int, Action, int], Dict[Reward, float]] = {}

        # 缺省：构建“最简确定性”环境（含 forbidden/target 的默认奖励）
        self._build_default_dynamics()

    # -----------------------------
    # 公共接口
    # -----------------------------
    def reset(self, start: Optional[Coord] = None) -> int:
        """重置环境，返回状态 id。"""
        self.s = start if start is not None else self.start
        return self.sid(self.s)

    def step(self, action: Optional[Action] = None) -> Tuple[int, Reward, bool, dict]:
        """
        单步交互：
          - 如 action is None：按 π(a|s) 抽样动作（若未设置 π，则均匀）
          - 然后按 P(s'|s,a) 与 P(r|s,a,s') 抽样转移与奖励
        返回: (next_state_id, reward, done, info)
        """
        sid = self.sid(self.s)
        if self._is_terminal(self.s):
            # 终止态保持吸收
            return sid, 0.0, True, {"terminal": True}

        a = action or self._sample_action(sid)
        next_sid = self._sample_next_state(sid, a)
        r = self._sample_reward(sid, a, next_sid)
        done = self._is_terminal(self.id2s[next_sid])

        self.s = self.id2s[next_sid]
        return next_sid, r, done, {"action": a}

    # ---------- 配置/编辑 ----------
    def set_policy_for_state(self, state: Coord, probs: Dict[Action, float]) -> None:
        """设置 π(a|s)。"""
        sid = self.sid(state)
        self._validate_prob_dict(probs, list(Action))
        self.policy[sid] = dict(probs)

    def set_transition(self, state: Coord, action: Action, next_probs: Dict[Coord, float]) -> None:
        """设置 P(s'|s,a)。"""
        sid = self.sid(state)
        next_sid_probs = {self.sid(ns): p for ns, p in next_probs.items()}
        self._validate_prob_dict(next_sid_probs, list(next_sid_probs.keys()))
        self.transitions[(sid, action)] = next_sid_probs

    def set_reward_distribution(
        self,
        state: Coord,
        action: Action,
        next_state: Coord,
        reward_probs: Dict[Reward, float],
    ) -> None:
        """设置 P(r|s,a,s')。"""
        sid, nsid = self.sid(state), self.sid(next_state)
        self._validate_prob_dict(reward_probs, list(reward_probs.keys()))
        self.rewards[(sid, action, nsid)] = dict(reward_probs)

    # ---------- 便捷接口 ----------
    def get_P(self) -> Dict[int, Dict[Action, List[Transition]]]:
        """
        返回 Gym 风格的 P[s][a] = List[(prob, next_sid, reward, done)]。
        对于未显式定义的 (s,a) 与奖励，按默认规则补齐。
        """
        P: Dict[int, Dict[Action, List[Transition]]] = {}
        for sid, s in enumerate(self.id2s):
            P[sid] = {}
            for a in Action.all():
                next_dist = self._get_transition_dist(sid, a)
                entries: List[Transition] = []
                for nsid, p in next_dist.items():
                    r_dist = self._get_reward_dist(sid, a, nsid)
                    done = self._is_terminal(self.id2s[nsid])
                    for r, pr in r_dist.items():
                        entries.append(Transition(prob=p * pr, next_state_id=nsid, reward=r, done=done))
                P[sid][a] = entries
        return P

    def allowed_actions(self, state: Coord) -> List[Action]:
        """终止态仅允许 STAY，其他态允许全部动作。"""
        return [Action.STAY] if self._is_terminal(state) else Action.all()

    # -----------------------------
    # 内部：默认规则与采样
    # -----------------------------
    def _build_default_dynamics(self) -> None:
        """
        缺省：确定性移动；撞墙则原地（若 stay_on_wall=True）。
        奖励：进入 forbidden=-10；到达 target=+1；其他=0。
        """
        for sid, s in enumerate(self.id2s):
            for a in Action.all():
                # 默认转移
                ns = self._move(s, a)
                nsid = self.sid(ns)
                self.transitions[(sid, a)] = {nsid: 1.0}

                # 默认奖励（确定性）
                default_r = self._default_reward(ns)
                self.rewards[(sid, a, nsid)] = {default_r: 1.0}

        # 默认策略：非终止态均匀
        for sid, s in enumerate(self.id2s):
            acts = self.allowed_actions(s)
            p = 1.0 / len(acts)
            self.policy[sid] = {a: p for a in acts}

    def _default_reward(self, ns: Coord) -> Reward:
        if self.target is not None and ns == self.target:
            return 1.0
        if ns in self.forbidden:
            return -10.0
        return 0.0

    def _move(self, s: Coord, a: Action) -> Coord:
        if self._is_terminal(s):
            return s  # 吸收

        r, c = s
        dr, dc = 0, 0
        if a == Action.UP:
            dr = -1
        elif a == Action.DOWN:
            dr = +1
        elif a == Action.LEFT:
            dc = -1
        elif a == Action.RIGHT:
            dc = +1
        elif a == Action.STAY:
            dr = dc = 0

        nr, nc = r + dr, c + dc
        if self._in_bounds((nr, nc)):
            return nr, nc

        # 撞墙
        return s if self.stay_on_wall else (max(0, min(self.h - 1, nr)), max(0, min(self.w - 1, nc)))

    def _sample_action(self, sid: int) -> Action:
        probs = self.policy.get(sid)
        if not probs:
            # 理论不会进来；兜底为均匀
            acts = Action.all()
            return self.rng.choice(acts)
        acts, ps = zip(*sorted(probs.items(), key=lambda kv: kv[0].value))
        return self.rng.choice(acts, p=np.array(ps, dtype=float))

    def _sample_next_state(self, sid: int, a: Action) -> int:
        dist = self._get_transition_dist(sid, a)
        nstates, ps = zip(*dist.items())
        return int(self.rng.choice(nstates, p=np.array(ps, dtype=float)))

    def _sample_reward(self, sid: int, a: Action, nsid: int) -> Reward:
        dist = self._get_reward_dist(sid, a, nsid)
        rs, ps = zip(*dist.items())
        return float(self.rng.choice(rs, p=np.array(ps, dtype=float)))

    # ---------- 取分布（含默认补齐） ----------
    def _get_transition_dist(self, sid: int, a: Action) -> Dict[int, float]:
        dist = self.transitions.get((sid, a))
        if dist is not None:
            return dist
        # 理论上 _build_default_dynamics 已填充；兜底：原地
        return {sid: 1.0}

    def _get_reward_dist(self, sid: int, a: Action, nsid: int) -> Dict[Reward, float]:
        dist = self.rewards.get((sid, a, nsid))
        if dist is not None:
            return dist
        # 兜底按默认规则
        ns = self.id2s[nsid]
        return {self._default_reward(ns): 1.0}

    # ---------- 工具 ----------
    def sid(self, s: Coord) -> int:
        return self.s2id[s]

    def _is_terminal(self, s: Coord) -> bool:
        if self.target is not None and s == self.target:
            return True
        if self.absorbing_forbidden and s in self.forbidden:
            return True
        return False

    def _in_bounds(self, s: Coord) -> bool:
        r, c = s
        return 0 <= r < self.h and 0 <= c < self.w

    @staticmethod
    def _validate_prob_dict(d: Dict, keys: List) -> None:
        total = float(sum(d.values()))
        if not keys:
            raise ValueError("空的概率字典。")
        if any(k not in d for k in keys if isinstance(keys, list) and len(keys) > 0):
            # 允许稀疏；仅要求提供的值总和为 1
            pass
        if not np.isclose(total, 1.0):
            raise ValueError(f"概率之和为 {total}，应为 1.0。")


