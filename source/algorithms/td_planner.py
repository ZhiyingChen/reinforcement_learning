# -*- coding: utf-8 -*-
# 路径：source/algorithms/td_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Iterable
import numpy as np

from source.utils.logger_manager import LoggerManager
from source.utils.timing import record_time_decorator
from source.utils.policy_ops import (
    sample_action_from_policy,
    greedy_one_hot_from_q,
    epsilon_greedy_from_q,
)
from source.domain_object import Action
from source.grid_world import GridWorld

Array = np.ndarray

@dataclass
class TDConfig:
    seed: int = 42
    log_dir: str = "logs/ch7"
    use_tensorboard: bool = True
    max_steps_per_episode: int = 10_000
    tie_breaker: Optional[List[Action]] = None

class TDPlanner:
    """
    Chapter 7: TD Control
      - SARSA
      - Expected SARSA
      - n-step SARSA
      - Q-learning (on-policy & off-policy)
    统一接口，复用 env.step() 在线更新（model-free）。
    """
    def __init__(self, env: GridWorld, cfg: Optional[TDConfig] = None) -> None:
        self.env = env
        self.cfg = cfg or TDConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.logger = LoggerManager(self.cfg.log_dir, self.cfg.use_tensorboard)
        self.gamma = getattr(env, "gamma", 0.99)
        self.nS = len(env.id2s)
        self.tie = self.cfg.tie_breaker or [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.STAY]
        self._episode_idx = 0

    # ---------- 公共工具 ----------
    def _allowed(self, sid: int) -> List[Action]:
        return self.env.allowed_actions(self.env.id2s[sid])

    def _epsilon_greedy_pi_from_Q(self, Q: Dict[int, Dict[Action, float]], epsilon: float) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed(sid)
            q_s = {a: Q.get(sid, {}).get(a, 0.0) for a in acts}
            pi[sid] = epsilon_greedy_from_q(q_s, acts, epsilon, self.tie)
        return pi

    def _greedy_pi_from_Q(self, Q: Dict[int, Dict[Action, float]]) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed(sid)
            q_s = {a: Q.get(sid, {}).get(a, -np.inf) for a in acts}
            pi[sid] = greedy_one_hot_from_q(q_s, self.tie)
        return pi

    def _uniform_pi(self) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed(sid)
            p = 1.0 / len(acts)
            pi[sid] = {a: p for a in acts}
        return pi

    # ---------- 采样一集（按给定策略） ----------
    def _run_episode(
        self,
        pi_behavior: Dict[int, Dict[Action, float]],
        max_steps: Optional[int] = None,
        start_state: Optional[int] = None,
    ) -> Tuple[List[int], List[Action], List[float]]:
        max_steps = max_steps or self.cfg.max_steps_per_episode
        if start_state is None:
            self.env.reset()
        else:
            self.env.reset(self.env.id2s[start_state])

        S: List[int] = []
        A: List[Action] = []
        R: List[float] = []

        sid = self.env.sid(self.env.s)
        # 按行为策略选首动作
        a = sample_action_from_policy(self.rng, pi_behavior[sid])

        for t in range(max_steps):
            S.append(sid); A.append(a)
            nsid, r, done, _ = self.env.step(a)
            R.append(float(r))
            sid = nsid
            if done:
                break
            a = sample_action_from_policy(self.rng, pi_behavior[sid])

        ep_ret = float(sum(R))
        self.logger.add_scalar("episode/return", ep_ret, self._episode_idx)
        self.logger.log(f"[EP {self._episode_idx}] len={len(S)} return={ep_ret:.3f}")
        self._episode_idx += 1
        return S, A, R

    # =========================================================
    #                     SARSA 系列
    # =========================================================
    @record_time_decorator("SARSA")
    def sarsa(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        min_epsilon: float = 0.01,
        pi_init: Optional[Dict[int, Dict[Action, float]]] = None,
        Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
        log_every: int = 20,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        SARSA: q_{t+1}(s,a) ← q_t(s,a) + α [ r + γ q_t(s',a') - q_t(s,a) ]
        on-policy：行为=目标=ε-greedy(Q)。
        """
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        pi = pi_init or self._epsilon_greedy_pi_from_Q(Q, epsilon)

        eps = float(epsilon)
        for ep in range(num_episodes):
            S, A, R = self._run_episode(pi)
            for t in range(len(S)):
                s, a = S[t], A[t]
                r = R[t]
                if t == len(S) - 1:
                    td_target = r
                else:
                    s_next, a_next = S[t + 1], A[t + 1]
                    q_next = Q.get(s_next, {}).get(a_next, 0.0)
                    td_target = r + self.gamma * q_next

                q_sa = Q.get(s, {}).get(a, 0.0)
                Q.setdefault(s, {})
                Q[s][a] = q_sa + alpha * (td_target - q_sa)

            # 策略改进（ε-greedy）
            pi = self._epsilon_greedy_pi_from_Q(Q, eps)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

            if (ep + 1) % log_every == 0:
                self.logger.add_scalar("SARSA/epsilon", eps, ep + 1)

        return Q, pi

    @record_time_decorator("ExpectedSARSA")
    def expected_sarsa(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        min_epsilon: float = 0.01,
        pi_init: Optional[Dict[int, Dict[Action, float]]] = None,
        Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
        log_every: int = 20,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        Expected SARSA:
          td_target = r + γ E_{a'~π(.|s')}[ q(s',a') ]
        相比 SARSA 用期望代替随机 a'，方差更低。
        """
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        pi = pi_init or self._epsilon_greedy_pi_from_Q(Q, epsilon)

        eps = float(epsilon)
        for ep in range(num_episodes):
            S, A, R = self._run_episode(pi)
            for t in range(len(S)):
                s, a = S[t], A[t]
                r = R[t]
                if t == len(S) - 1:
                    td_target = r
                else:
                    s_next = S[t + 1]
                    # 期望 q(s',a') = sum_a π(a|s') * q(s',a)
                    exp_q = 0.0
                    for a2, p in pi[s_next].items():
                        exp_q += p * Q.get(s_next, {}).get(a2, 0.0)
                    td_target = r + self.gamma * exp_q

                q_sa = Q.get(s, {}).get(a, 0.0)
                Q.setdefault(s, {})
                Q[s][a] = q_sa + alpha * (td_target - q_sa)

            # ε-greedy 策略更新
            pi = self._epsilon_greedy_pi_from_Q(Q, eps)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)
            if (ep + 1) % log_every == 0:
                self.logger.add_scalar("ExpectedSARSA/epsilon", eps, ep + 1)

        return Q, pi

    @record_time_decorator("nStepSARSA")
    def n_step_sarsa(
        self,
        *,
        n: int = 3,
        num_episodes: int = 2000,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        min_epsilon: float = 0.01,
        Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        n-step SARSA（前视）：G_t^(n) = r_{t+1}+...+γ^{n} q(s_{t+n}, a_{t+n})
        含 Sarsa(n=1) 与 MC(n=∞) 两极。更新在 t+n 时对 (s_t,a_t) 执行。
        """
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        eps = float(epsilon)
        pi = self._epsilon_greedy_pi_from_Q(Q, eps)

        for ep in range(num_episodes):
            S, A, R = self._run_episode(pi)
            T = len(S)
            # 累积 n-step return 的窗口更新
            for t in range(T):
                # 回看窗口 [t, t+n]
                horizon = min(n, T - t - 1)  # 还剩的真正可累加步数（除去bootstrap一步）
                G = 0.0
                for k in range(horizon):
                    G += (self.gamma ** k) * R[t + k]
                # Bootstrap
                if t + horizon < T - 0:
                    s_boot = S[t + horizon]
                    a_boot = A[t + horizon]  # Sarsa 的 bootstrapping 使用经历中的 a_{t+n}
                    G += (self.gamma ** horizon) * Q.get(s_boot, {}).get(a_boot, 0.0)

                s, a = S[t], A[t]
                q_sa = Q.get(s, {}).get(a, 0.0)
                Q.setdefault(s, {})
                Q[s][a] = q_sa + alpha * (G - q_sa)

            # ε-greedy 更新
            pi = self._epsilon_greedy_pi_from_Q(Q, eps)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

        return Q, pi

    # =========================================================
    #                     Q-learning
    # =========================================================
    @record_time_decorator("Q-Learning-OnPolicy")
    def q_learning_on_policy(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        min_epsilon: float = 0.01,
        Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        On-policy 版本：行为=目标=ε-greedy(Q)，但目标使用 max_a q(s',a) 的 off-policy 目标；
        算法表达式：
          q(s,a) ← q(s,a) + α [ r + γ max_a' q(s',a') - q(s,a) ]。
        """
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        eps = float(epsilon)
        pi = self._epsilon_greedy_pi_from_Q(Q, eps)

        for ep in range(num_episodes):
            S, A, R = self._run_episode(pi)
            for t in range(len(S)):
                s, a = S[t], A[t]
                r = R[t]
                if t == len(S) - 1:
                    td_target = r
                else:
                    s_next = S[t + 1]
                    # 目标使用 greedy
                    q_next = max(Q.get(s_next, {}).values()) if Q.get(s_next) else 0.0
                    td_target = r + self.gamma * q_next
                q_sa = Q.get(s, {}).get(a, 0.0)
                Q.setdefault(s, {})
                Q[s][a] = q_sa + alpha * (td_target - q_sa)
            # 行为=目标=ε-greedy
            pi = self._epsilon_greedy_pi_from_Q(Q, eps)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

        return Q, pi

    @record_time_decorator("Q-Learning-OffPolicy")
    def q_learning_off_policy(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.1,
        behavior_epsilon: float = 0.3,   # 行为策略更探索
        target_epsilon: float = 0.05,    # 目标策略接近 greedy
        Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        """
        经典 off-policy：行为策略 π_b ≠ 目标策略 π_t。
        采样用 π_b（高 ε），更新目标使用 max_a q(s',a)；目标策略维持接近 greedy(Q)。
        """
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        pi_t = self._epsilon_greedy_pi_from_Q(Q, target_epsilon)  # 目标策略
        pi_b = self._epsilon_greedy_pi_from_Q(Q, behavior_epsilon)  # 行为策略

        for ep in range(num_episodes):
            # 用行为策略采样
            S, A, R = self._run_episode(pi_b)
            for t in range(len(S)):
                s, a = S[t], A[t]
                r = R[t]
                if t == len(S) - 1:
                    td_target = r
                else:
                    s_next = S[t + 1]
                    q_next = max(Q.get(s_next, {}).values()) if Q.get(s_next) else 0.0
                    td_target = r + self.gamma * q_next
                q_sa = Q.get(s, {}).get(a, 0.0)
                Q.setdefault(s, {})
                Q[s][a] = q_sa + alpha * (td_target - q_sa)

            # 更新目标策略为近似 greedy(Q)，行为策略保持探索性
            pi_t = self._epsilon_greedy_pi_from_Q(Q, target_epsilon)
            pi_b = self._epsilon_greedy_pi_from_Q(Q, behavior_epsilon)

        return Q, pi_t