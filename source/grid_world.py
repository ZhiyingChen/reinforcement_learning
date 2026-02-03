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
        reward_step: float = 0.0,
        reward_boundary: float = 0.0,
        reward_forbidden: float = -1.0,
        reward_target: float = 10.0,
        gamma: float = 0.9,

    ):
        assert height > 0 and width > 0
        self.h, self.w = height, width
        self.forbidden: set[Coord] = set(forbidden or [])
        self.target: Optional[Coord] = target
        self.start: Coord = start
        self.absorbing_forbidden = absorbing_forbidden
        self.stay_on_wall = stay_on_wall
        self.reward_step = reward_step
        self.reward_boundary = reward_boundary
        self.reward_forbidden = reward_forbidden
        self.reward_target = reward_target
        self.gamma = gamma

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

    def step(self, action=None):
        sid = self.sid(self.s)
        if self._is_terminal(self.s):
            return sid, 0.0, True, {"terminal": True}

        a = action or self._sample_action(sid)
        next_state, hit_wall = self._move(self.s, a)
        next_sid = self.sid(next_state)

        # 使用新的奖励逻辑
        r = self._compute_reward(self.s, a, next_state, hit_wall)

        done = self._is_terminal(next_state)
        self.s = next_state
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
                ns, hit_wall = self._move(s, a)
                nsid = self.sid(ns)
                self.transitions[(sid, a)] = {nsid: 1.0}

                # 默认奖励（确定性）
                default_r = self._default_reward(s=s, a=a, ns=ns)
                self.rewards[(sid, a, nsid)] = {default_r: 1.0}

        # 默认策略：非终止态均匀
        for sid, s in enumerate(self.id2s):
            acts = self.allowed_actions(s)
            p = 1.0 / len(acts)
            self.policy[sid] = {a: p for a in acts}

    def _default_reward(self,s: Coord, a: Action, ns: Coord) -> Reward:
        # target reward
        if self.target is not None and ns == self.target:
            return float(self.reward_target)

        # forbidden reward
        if ns in self.forbidden:
            return float(self.reward_forbidden)

        # boundary reward：判断是否撞墙（原地不动）
        # 若 stay_on_wall=True 且移动后 ns == 当前 s，则视为撞墙
        if a != Action.STAY and ns == s:
            return float(self.reward_boundary)

        return float(self.reward_step)

    def _compute_reward(self, s, a, ns, hit_wall):
        if hit_wall:
            return self.reward_boundary

        if ns == self.target:
            return self.reward_target

        if ns in self.forbidden:
            return self.reward_forbidden

        return self.reward_step  # 新增普通格子 reward（如你需要）

    def _move(self, s: Coord, a: Action):
        r, c = s
        dr, dc = 0, 0
        if a == Action.UP:
            dr = -1
        elif a == Action.DOWN:
            dr = 1
        elif a == Action.LEFT:
            dc = -1
        elif a == Action.RIGHT:
            dc = 1
        elif a == Action.STAY:
            dr = dc = 0  # stay 不算撞墙

        nr, nc = r + dr, c + dc

        # 合法移动
        if self._in_bounds((nr, nc)):
            return (nr, nc), False

        # 撞墙（无论 stay_on_wall=True 或 False 都算撞墙）
        if self.stay_on_wall:
            return s, True
        else:
            clipped = (
                max(0, min(self.h - 1, nr)),
                max(0, min(self.w - 1, nc)),
            )
            return clipped, True

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
        s = self.id2s[sid]
        ns = self.id2s[nsid]
        return {self._default_reward(s, a, ns): 1.0}

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


