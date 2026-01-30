# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np

from source.grid_world import GridWorld
from source.domain_object import Action
from source.utils.policy_ops import (
    sample_action_from_policy, greedy_one_hot_from_q, epsilon_greedy_from_q
)
from source.utils.logger_manager import LoggerManager
from source.utils.timing import record_time_decorator

@dataclass
class MCConfig:
    gamma: Optional[float] = None           # None -> 取 env.gamma，否则默认 0.9
    first_visit: bool = True                # Basic/ES 默认 first-visit；ε-greedy 常用 every-visit
    max_steps_per_episode: int = 10_000     # Episode 上限，防止意外循环
    tie_breaker: Optional[List[Action]] = None
    seed: Optional[int] = 42

class MCPlanner:
    """
    Monte-Carlo Control on GridWorld
    - MC Basic（基于已有 π，按状态-动作对采集若干 episode，取均值评估 Q，再贪心改进）
    - MC Exploring Starts（ES）：每个 episode 随机选择起始 (s0, a0)，first-visit 更新
    - MC ε-Greedy（On-policy）：行为=目标=ε-greedy(Q)，every-visit 更新
    """

    def __init__(self, env: GridWorld, cfg: Optional[MCConfig] = None, log_dir="logs/mc", use_tb=True):
        self.env = env
        self.cfg = cfg or MCConfig()
        self.logger = LoggerManager(log_dir, use_tensorboard=use_tb)
        self.logger.log("MCPlanner initialized.")
        self.episode_counter = 0

        self.gamma = (
            getattr(env, "gamma", 0.9) if self.cfg.gamma is None else float(self.cfg.gamma)
        )
        self.tie_breaker = self.cfg.tie_breaker or [
            Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.STAY
        ]
        self.rng = np.random.default_rng(self.cfg.seed)

        # 状态总数
        self.nS = len(self.env.id2s)

        # 预先缓存：所有可起始（非终止）状态-动作对（用于 ES / 采样）
        self._all_pairs: List[Tuple[int, Action]] = self._enumerate_state_action_pairs()

    # ------------------------------------------------------------------ #
    # 公共：episode 生成与回报计算
    # ------------------------------------------------------------------ #
    def generate_episode(
        self,
        pi: Dict[int, Dict[Action, float]],
        *,
        start_state: Optional[int] = None,
        start_action: Optional[Action] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[List[int], List[Action], List[float]]:
        """
        生成一条 episode（基于 env.step），返回 (S, A, R)
        - 若给出 (start_state, start_action)，则强制第一步从该对开始（ES / Basic）
        - 否则：从 env.start 开始，按 π 采样动作（On-policy）
        """
        max_steps = max_steps or self.cfg.max_steps_per_episode

        # 设定起点
        if start_state is not None:
            s_coord = self.env.id2s[start_state]
            self.env.reset(start=s_coord)
        else:
            self.env.reset()

        S: List[int] = []
        A: List[Action] = []
        R: List[float] = []

        sid = self.env.sid(self.env.s)

        # 第一动作处理（ES 或 Basic 从 (s0,a0) 开始）
        if start_action is not None:
            a = start_action
        else:
            # 从策略采样
            pi_s = pi.get(sid, {})
            if not pi_s:
                # 兜底：均匀随机允许动作
                acts = self._allowed_actions_sid(sid)
                p = 1.0 / len(acts)
                pi_s = {a_: p for a_ in acts}
            a = sample_action_from_policy(self.rng, pi_s)

        for t in range(max_steps):
            S.append(sid)
            A.append(a)
            next_sid, r, done, _info = self.env.step(a)
            R.append(float(r))
            sid = next_sid
            if done:
                break
            # 下一步动作仍按策略采样
            pi_s = pi.get(sid, {})
            if not pi_s:
                acts = self._allowed_actions_sid(sid)
                p = 1.0 / len(acts)
                pi_s = {a_: p for a_ in acts}
            a = sample_action_from_policy(self.rng, pi_s)

        total_reward = sum(R)
        self.logger.add_scalar("episode/return", total_reward, self.episode_counter)
        self.logger.log(f"[Episode {self.episode_counter}] length={len(S)}, total_reward={total_reward}")
        self.episode_counter += 1
        return S, A, R

    def returns_from_rewards(self, rewards: List[float]) -> List[float]:
        """G_t 累积回报（从后往前计算）"""
        G = 0.0
        out = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            out[t] = G
        return out

    # ------------------------------------------------------------------ #
    # 算法 1：MC Basic（model-free variant of policy iteration）
    # ------------------------------------------------------------------ #
    @record_time_decorator("MC BASIC")
    def mc_basic(
        self,
        *,
        init_policy: Optional[Dict[int, Dict[Action, float]]] = None,
        episodes_per_pair: int = 4,
        outer_iters: int = 5,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        返回: (Q, π*)
        - 每一外层迭代：对每个 (s,a) 采集若干 episode，平均回报估计 q(s,a)，再贪心改进策略
        - 适合小状态空间（如 GridWorld）；大空间可改“按需子集采集”
        """

        # 初始化策略（均匀）
        pi = init_policy or self._uniform_policy()

        # 初始化 Q、计数与累计（增量平均）
        Q: Dict[int, Dict[Action, float]] = {s: {} for s in range(self.nS)}
        returns_sum: Dict[Tuple[int, Action], float] = {}
        returns_count: Dict[Tuple[int, Action], int] = {}

        for iter_id in range(outer_iters):
            self.logger.log(f"--- MC Basic Iter {iter_id} ---")
            self.logger.add_scalar("mc_basic/outer_iter", iter_id, iter_id)

            for (sid, a0) in self._all_pairs:
                for _e in range(episodes_per_pair):
                    S, A, R = self.generate_episode(pi, start_state=sid, start_action=a0)
                    Gs = self.returns_from_rewards(R)

                    key = (sid, a0)
                    returns_sum[key] = returns_sum.get(key, 0.0) + Gs[0]
                    returns_count[key] = returns_count.get(key, 0) + 1
                    Q.setdefault(sid, {})
                    Q[sid][a0] = returns_sum[key] / returns_count[key]

            # 策略改进（贪心）
            pi = self._greedy_policy_from_Q(Q)
            self.logger.log("Policy improved for MC Basic.")

        return Q, pi

    # ------------------------------------------------------------------ #
    # 算法 2：MC Exploring Starts（每集随机起始 (s0, a0)）
    # ------------------------------------------------------------------ #
    @record_time_decorator("MC Exploring Starts")
    def mc_exploring_starts(
        self,
        *,
        num_episodes: int = 10_000,
        init_policy: Optional[Dict[int, Dict[Action, float]]] = None,
        first_visit: Optional[bool] = None,  # 默认 True
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        返回: (Q, π*)
        - 每个 episode：随机均匀选取起始 (s0, a0)，随后按当前 π 生成一条 episode
        - first-visit 更新，提高无偏性
        """
        first_visit = self.cfg.first_visit if first_visit is None else first_visit

        pi = init_policy or self._uniform_policy()
        Q: Dict[int, Dict[Action, float]] = {s: {} for s in range(self.nS)}
        returns_sum: Dict[Tuple[int, Action], float] = {}
        returns_count: Dict[Tuple[int, Action], int] = {}

        for ep in range(num_episodes):
            s0, a0 = self._uniform_random_pair()
            self.logger.log(f"[ES Episode {ep}] start=({s0},{a0})")

            S, A, R = self.generate_episode(pi, start_state=s0, start_action=a0)
            Gs = self.returns_from_rewards(R)

            visited: set[Tuple[int, Action]] = set()
            for t, (s_t, a_t) in enumerate(zip(S, A)):
                key = (s_t, a_t)
                if first_visit and key in visited:
                    continue
                visited.add(key)

                Gt = Gs[t]
                returns_sum[key] = returns_sum.get(key, 0.0) + Gt
                returns_count[key] = returns_count.get(key, 0) + 1
                Q.setdefault(s_t, {})
                Q[s_t][a_t] = returns_sum[key] / returns_count[key]

            # 每集后进行一次贪心改进
            pi = self._greedy_policy_from_Q(Q)
            self.logger.add_scalar("mc_es/episode_return", Gs[0], ep)

        return Q, pi

    # ------------------------------------------------------------------ #
    # 算法 3：MC ε-Greedy（On-policy MC Control）
    # ------------------------------------------------------------------ #
    @record_time_decorator("MC Epsilon Greedy")
    def mc_epsilon_greedy(
        self,
        *,
        num_episodes: int = 10_000,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,   # 例如 0.999
        min_epsilon: float = 0.01,
        every_visit: bool = True,                # On-policy 常用 every-visit
        init_Q: Optional[Dict[int, Dict[Action, float]]] = None,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        返回: (Q, π_ε*)  —— 行为=目标=ε-greedy(Q)，保证持续探索
        - 每集用当前 ε-greedy 策略生成 episode
        - 使用 every-visit（默认）或 first-visit 更新 Q
        - 每集末根据新 Q 更新整张 ε-greedy 策略（on-policy）
        """
        # 初始化 Q 与计数器
        Q: Dict[int, Dict[Action, float]] = init_Q or {s: {} for s in range(self.nS)}
        returns_sum: Dict[Tuple[int, Action], float] = {}
        returns_count: Dict[Tuple[int, Action], int] = {}

        # 初始策略（ε-greedy on Q）
        pi = self._epsilon_greedy_policy_from_Q(Q, epsilon)

        eps = float(epsilon)
        for episode in range(num_episodes):

            # 用 π_ε 采样一条 episode（起点 env.start）
            S, A, R = self.generate_episode(pi)
            Gs = self.returns_from_rewards(R)

            self.logger.add_scalar("mc_eps/epsilon", eps, episode)
            self.logger.add_scalar("mc_eps/episode_return", Gs[0], episode)

            visited: set[Tuple[int, Action]] = set()
            for t, (s_t, a_t) in enumerate(zip(S, A)):
                key = (s_t, a_t)
                if (not every_visit) and key in visited:
                    continue
                visited.add(key)

                Gt = Gs[t]
                returns_sum[key] = returns_sum.get(key, 0.0) + Gt
                returns_count[key] = returns_count.get(key, 0) + 1
                Q.setdefault(s_t, {})
                Q[s_t][a_t] = returns_sum[key] / returns_count[key]

            # 策略改进：新的 ε-greedy(Q)
            pi = self._epsilon_greedy_policy_from_Q(Q, eps)
            self.logger.log(f"[ε-Greedy] eps={eps:.4f}")

            # ε 衰减（可选）
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

        return Q, pi

    # ------------------------------------------------------------------ #
    # 辅助：策略构造
    # ------------------------------------------------------------------ #
    def _uniform_policy(self) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed_actions_sid(sid)
            if not acts:
                pi[sid] = {Action.STAY: 1.0}
            else:
                p = 1.0 / len(acts)
                pi[sid] = {a: p for a in acts}
        return pi

    def _greedy_policy_from_Q(self, Q: Dict[int, Dict[Action, float]]) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed_actions_sid(sid)
            q_s = {a: Q.get(sid, {}).get(a, -np.inf) for a in acts}
            pi[sid] = greedy_one_hot_from_q(q_s, self.tie_breaker)
        return pi

    def _epsilon_greedy_policy_from_Q(self, Q: Dict[int, Dict[Action, float]], epsilon: float) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed_actions_sid(sid)
            q_s = {a: Q.get(sid, {}).get(a, 0.0) for a in acts}
            pi[sid] = epsilon_greedy_from_q(q_s, acts, epsilon, self.tie_breaker)
        return pi

    # ------------------------------------------------------------------ #
    # 辅助：状态-动作对枚举与采样
    # ------------------------------------------------------------------ #
    def _allowed_actions_sid(self, sid: int) -> List[Action]:
        s = self.env.id2s[sid]
        return self.env.allowed_actions(s)

    def _enumerate_state_action_pairs(self) -> List[Tuple[int, Action]]:
        pairs: List[Tuple[int, Action]] = []
        for sid in range(self.nS):
            for a in self._allowed_actions_sid(sid):
                # 终止态只允许 STAY；ES 中可以选择 (s_terminal, STAY) 但通常意义不大
                pairs.append((sid, a))
        return pairs

    def _uniform_random_pair(self) -> Tuple[int, Action]:
        return self.rng.choice(self._all_pairs)
