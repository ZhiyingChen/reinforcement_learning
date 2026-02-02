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
    # ---------- SARSA（一步一更新） ----------
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
            max_steps_per_episode: Optional[int] = None,
            diag_every: int = 100,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        eps = float(epsilon)
        pi = pi_init or self._epsilon_greedy_pi_from_Q(Q, eps)

        for ep in range(num_episodes):
            # 初始化一集
            self.env.reset()
            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, pi[sid])
            ep_ret = 0.0

            for t in range(self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode):
                # 与环境交互一步
                nsid, r, done, _ = self.env.step(a)
                ep_ret += r

                # 先用 π_t 在 s' 选择 a' （课件顺序）
                if not done:
                    a_next = sample_action_from_policy(self.rng, pi[nsid])

                # 计算 TD 目标并更新 Q(s,a)
                q_sa = Q.get(sid, {}).get(a, 0.0)
                if done:
                    td_target = r
                else:
                    q_next = Q.get(nsid, {}).get(a_next, 0.0)
                    td_target = r + self.gamma * q_next
                Q.setdefault(sid, {})
                delta = td_target - q_sa  # ★ TD error
                Q[sid][a] = q_sa + alpha * delta

                # 立刻对当前状态做策略改进（ε-greedy(Q)）
                acts = self._allowed(sid)
                q_s = {aa: Q[sid].get(aa, -np.inf) for aa in acts}
                pi[sid] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

                # --- 诊断日志（按步间隔写） ---
                if t % diag_every == 0:

                    self.logger.add_scalar("SARSA/td_error_abs", abs(float(delta)), ep * 10_000 + t)
                    self.logger.add_scalar("SARSA/q_max_s", max(q_s.values()), ep * 10_000 + t)
                    # 可选：写到 run.log（不建议太频繁）
                    self.logger.log(f"[SARSA] ep={ep} t={t} |δ|={abs(float(delta)):.3e}")

                if done:
                    break
                # 滚动到下一步（注意：a_next 是基于 π_t 选出的）
                sid, a = nsid, a_next

            # 每集：回报 + ε
            self.logger.add_scalar("SARSA/epsilon", eps, ep)
            self.logger.add_scalar("episode/return", ep_ret, ep)

            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)
                # 同步刷新整张策略的 ε（可选，保持行为与目标一致）
                for s in range(self.nS):
                    acts = self._allowed(s)
                    q_s = {aa: Q.get(s, {}).get(aa, 0.0) for aa in acts}
                    pi[s] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

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
            Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
            max_steps_per_episode: Optional[int] = None,
            diag_every: int = 100,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:

        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        eps = float(epsilon)
        pi = self._epsilon_greedy_pi_from_Q(Q, eps)

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, pi[sid])
            ep_ret = 0.0

            for t in range(self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode):
                nsid, r, done, _ = self.env.step(a)
                ep_ret += r

                # Expected target：E_{a'~π(s')} q(s',a')
                if done:
                    td_target = r
                else:
                    exp_q = 0.0
                    for a2, p in pi[nsid].items():
                        exp_q += p * Q.get(nsid, {}).get(a2, 0.0)
                    td_target = r + self.gamma * exp_q

                q_sa = Q.get(sid, {}).get(a, 0.0)
                Q.setdefault(sid, {})
                delta = td_target - q_sa
                Q[sid][a] = q_sa + alpha * delta

                # 只对 s_t 进行策略改进
                acts = self._allowed(sid)
                q_s = {aa: Q[sid].get(aa, -np.inf) for aa in acts}
                pi[sid] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

                if t % diag_every == 0:
                    self.logger.add_scalar("ExpectedSARSA/td_error_abs", abs(float(delta)), ep * 10_000 + t)
                    self.logger.add_scalar("ExpectedSARSA/q_max_s", max(q_s.values()), ep * 10_000 + t)
                    # 可选：写到 run.log（不建议太频繁）
                    self.logger.log(f"[ExpectedSARSA] ep={ep} t={t} |δ|={abs(float(delta)):.3e}")

                if done:
                    break
                # 下一步动作先按“更新前”的策略 π_t(s') 选择
                a_next = sample_action_from_policy(self.rng, pi[nsid])
                sid, a = nsid, a_next

            self.logger.add_scalar("episode/return", ep_ret, ep)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)
                for s in range(self.nS):
                    acts = self._allowed(s)
                    q_s = {aa: Q.get(s, {}).get(aa, 0.0) for aa in acts}
                    pi[s] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

        return Q, pi

    # ---------- n-step SARSA（在线前视，更新即改策略） ----------
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
            max_steps_per_episode: Optional[int] = None,
            diag_every: int = 100,  # ★
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        assert n >= 1
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        eps = float(epsilon)
        pi = self._epsilon_greedy_pi_from_Q(Q, eps)

        for ep in range(num_episodes):
            self.env.reset()
            S: List[int] = []
            A: List[Action] = []
            R: List[float] = [0.0]  # 方便 1-based 对齐，R[t+1] 对应本步回报

            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, pi[sid])
            S.append(sid)
            A.append(a)
            ep_ret = 0.0
            T = np.inf
            t = 0

            while True:
                if t < (self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode):
                    if t + 1 < T:
                        nsid, r, done, _ = self.env.step(A[t])
                        ep_ret += r
                        S_next = nsid
                        R.append(float(r))
                        if done:
                            T = t + 1
                        else:
                            a_next = sample_action_from_policy(self.rng, pi[S_next])
                            S.append(S_next)
                            A.append(a_next)

                    tau = t - n + 1
                    if tau >= 0:
                        # 计算 G_tau^(n)
                        G = 0.0
                        upper = min(tau + n, T)
                        for i in range(tau + 1, upper + 1):
                            G += (self.gamma ** (i - tau - 1)) * R[i]
                        if tau + n < T:
                            # bootstrap with q(S_{tau+n}, A_{tau+n})
                            s_boot, a_boot = S[tau + n], A[tau + n]
                            G += (self.gamma ** n) * Q.get(s_boot, {}).get(a_boot, 0.0)

                        # 在线更新 Q，并立刻只改 S_tau 的策略
                        s_tau, a_tau = S[tau], A[tau]
                        q_old = Q.get(s_tau, {}).get(a_tau, 0.0)
                        Q.setdefault(s_tau, {})
                        delta = G - q_old
                        Q[s_tau][a_tau] = q_old + alpha * delta

                        acts = self._allowed(s_tau)
                        q_s = {aa: Q[s_tau].get(aa, -np.inf) for aa in acts}
                        pi[s_tau] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

                        # 诊断
                        if t % diag_every == 0:
                            self.logger.add_scalar(f"nStepSARSA(n={n})/td_error_abs", abs(float(delta)), ep * 10_000 + t)
                            self.logger.add_scalar(f"nStepSARSA(n={n})/q_max_s", max(q_s.values()), ep * 10_000 + t)
                            self.logger.log(f"[nStepSARSA] ep={ep} t={t} |δ|={abs(float(delta)):.3e}")

                    t += 1
                    if tau >= T - 1:
                        break

            self.logger.add_scalar("episode/return", ep_ret, ep)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)
                for s in range(self.nS):
                    acts = self._allowed(s)
                    q_s = {aa: Q.get(s, {}).get(aa, 0.0) for aa in acts}
                    pi[s] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

        return Q, pi

    # =========================================================
    #                     Q-learning
    # =========================================================
    # ---------- Q-learning（on-policy，一步一更新） ----------
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
            max_steps_per_episode: Optional[int] = None,
            diag_every: int = 100,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        eps = float(epsilon)
        pi = self._epsilon_greedy_pi_from_Q(Q, eps)

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, pi[sid])
            ep_ret = 0.0

            for t in range(self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode):
                nsid, r, done, _ = self.env.step(a)
                ep_ret += r

                # Q-learning 目标：max_a' q(s',a')
                if done:
                    td_target = r
                else:
                    q_next = max(Q.get(nsid, {}).values()) if Q.get(nsid) else 0.0
                    td_target = r + self.gamma * q_next

                q_sa = Q.get(sid, {}).get(a, 0.0)
                Q.setdefault(sid, {})
                delta = td_target - q_sa
                Q[sid][a] = q_sa + alpha * delta

                # 立刻只改 s_t 的策略（on-policy 版本依旧保持 ε-greedy）
                acts = self._allowed(sid)
                q_s = {aa: Q[sid].get(aa, -np.inf) for aa in acts}
                pi[sid] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

                if t % diag_every == 0:
                    self.logger.add_scalar("Q-On/td_error_abs", abs(float(delta)), ep * 10_000 + t)
                    self.logger.add_scalar("Q-On/q_max_s", max(q_s.values()), ep * 10_000 + t)

                if done:
                    break
                # 下一步 a_{t+1} 仍从“更新前”的策略 π_t(s') 采样（课件顺序）
                a_next = sample_action_from_policy(self.rng, pi[nsid])
                sid, a = nsid, a_next

            self.logger.add_scalar("episode/return", ep_ret, ep)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)
                for s in range(self.nS):
                    acts = self._allowed(s)
                    q_s = {aa: Q.get(s, {}).get(aa, 0.0) for aa in acts}
                    pi[s] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

        return Q, pi

    # ---------- Q-learning（off-policy：π_b 采样，π_t 近 greedy；一步一更新） ----------
    @record_time_decorator("Q-Learning-OffPolicy")
    def q_learning_off_policy(
            self,
            *,
            num_episodes: int = 2000,
            alpha: float = 0.1,
            behavior_epsilon: float = 0.3,
            target_epsilon: float = 0.05,
            Q_init: Optional[Dict[int, Dict[Action, float]]] = None,
            max_steps_per_episode: Optional[int] = None,
            diag_every: int = 100,
    ) -> Tuple[Dict[int, Dict[Action, float]], Dict[int, Dict[Action, float]]]:
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}
        pi_t = self._epsilon_greedy_pi_from_Q(Q, target_epsilon)  # 目标策略（展示/评估）
        pi_b = self._epsilon_greedy_pi_from_Q(Q, behavior_epsilon)  # 行为策略（采样）

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, pi_b[sid])
            ep_ret = 0.0

            for t in range(self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode):
                nsid, r, done, _ = self.env.step(a)
                ep_ret += r

                if done:
                    td_target = r
                else:
                    q_next = max(Q.get(nsid, {}).values()) if Q.get(nsid) else 0.0
                    td_target = r + self.gamma * q_next

                q_sa = Q.get(sid, {}).get(a, 0.0)
                Q.setdefault(sid, {})
                delta = td_target - q_sa
                Q[sid][a] = q_sa + alpha * delta

                # 只改 s_t 的 π_t（接近 greedy）；π_b 保持较大 ε 继续探索
                acts = self._allowed(sid)
                q_s = {aa: Q[sid].get(aa, -np.inf) for aa in acts}
                pi_t[sid] = epsilon_greedy_from_q(q_s, acts, target_epsilon, self.tie)
                pi_b[sid] = epsilon_greedy_from_q(q_s, acts, behavior_epsilon, self.tie)

                if t % diag_every == 0:
                    self.logger.add_scalar("Q-Off/td_error_abs", abs(float(delta)), ep * 10_000 + t)
                    self.logger.add_scalar("Q-Off/q_max_s", max(q_s.values()), ep * 10_000 + t)

                if done:
                    break
                # 下一步行动来自“行为策略” π_b
                a_next = sample_action_from_policy(self.rng, pi_b[nsid])
                sid, a = nsid, a_next

            self.logger.add_scalar("episode/return", ep_ret, ep)

        return Q, pi_t