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

    # ---------- 采样一集（按给定策略） ----------
    def _policy_improve_at_state(self, pi, Q, sid: int, epsilon: float) -> Dict[int, Dict[Action, float]]:
        """仅对给定状态 sid 做一次 ε-greedy 政策改进（就地更新 pi）。"""
        acts = self._allowed(sid)
        # 若该状态还没有任何 Q 值，用 -inf 使得已出现过的动作优先
        q_s = {a: Q.get(sid, {}).get(a, -np.inf) for a in acts}
        pi[sid] = epsilon_greedy_from_q(q_s, acts, epsilon, self.tie)
        return pi

    # =========================================================
    #                     SARSA 系列
    # =========================================================
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

            done = False
            t = 0
            while ((not done) and
                   (t < self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode)):
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

                t += 1
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

    # --- 用原有函数名做兼容封装 ----------------------------------------
    # ------------------- SARSA = SARSA(n) 的 n=1 版 ----------------------
    @record_time_decorator("SARSA")
    def sarsa(
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
    ):
        return self.n_step_sarsa(
            n=1,
            num_episodes=num_episodes,
            alpha=alpha,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            Q_init=Q_init,
            max_steps_per_episode=max_steps_per_episode,
            diag_every=diag_every
        )

    # ---------------- 重写：按“片段式（segment）”实现的 n-step SARSA -------------
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
            diag_every: int = 100,
    ):
        """
        片段式 n-step SARSA（前视 n 步；不足 n 步则在终止处截断）。
        每次仅生成一个长度<=n的片段，用该片段一次性更新 Q(S_t, A_t)，
        然后对 S_t 做一次 ε-greedy 政策改进；之后将环境“前进 1 步”，
        从新状态开始生成下一段片段。episode 的结束由 env 的终止判定决定。
        """
        assert n >= 1
        Q: Dict[int, Dict[Action, float]] = Q_init or {s: {} for s in range(self.nS)}

        eps = float(epsilon)
        # 行为/目标统一为 on-policy 的 ε-greedy(Q)
        pi = {}
        for sid in range(self.nS):
            acts = self._allowed(sid)
            q_s = {a: Q.get(sid, {}).get(a, 0.0) for a in acts}
            pi[sid] = epsilon_greedy_from_q(q_s, acts, eps, self.tie)

        max_steps = max_steps_per_episode or self.cfg.max_steps_per_episode

        for ep in range(num_episodes):
            # --- 初始化 episode ---
            self.env.reset()
            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, pi[sid])

            ep_ret = 0.0
            step_counter = 0
            done = False

            while not done and step_counter < max_steps:
                # ===== (1) 从当前 (sid, a) 出发，生成“至多 n 步”的 SARSA 片段 =====
                S: List[int] = [sid]
                A: List[Action] = [a]
                R: List[float] = [0.0]  # 让 R[t+1] 对齐本步回报
                seg_len = 0

                cur_sid, cur_a = sid, a
                seg_done = False

                while seg_len < n and not seg_done:
                    nsid, r, done_step, _ = self.env.step(cur_a)
                    ep_ret += float(r)
                    R.append(float(r))
                    seg_len += 1

                    if done_step:
                        seg_done = True  # 片段提前在终止处截断
                        done = True  # 整个 episode 也结束
                        break
                    else:
                        # 还没终止：按当前策略抽样下一动作并扩展片段
                        a_next = sample_action_from_policy(self.rng, pi[nsid])
                        S.append(nsid)
                        A.append(a_next)
                        cur_sid, cur_a = nsid, a_next

                # ===== (2) 用片段一次性计算 G，并更新 Q(S_t, A_t) =====
                # 片段的“起点”就是 (sid, a)，对应 S[0], A[0]
                # G = sum_{i=1..seg_len} gamma^{i-1} * R[i] + (若未终止且 seg_len==n) gamma^n * Q(S_n, A_n)
                G = 0.0
                for i in range(1, seg_len + 1):
                    G += (self.gamma ** (i - 1)) * R[i]

                if (not seg_done) and (seg_len == n):
                    # bootstrap with Q(S_n, A_n)
                    s_boot, a_boot = S[-1], A[-1]
                    G += (self.gamma ** n) * Q.get(s_boot, {}).get(a_boot, 0.0)

                q_old = Q.get(sid, {}).get(a, 0.0)
                Q.setdefault(sid, {})
                delta = G - q_old
                Q[sid][a] = q_old + alpha * delta

                # ===== (3) 仅在 S_t 做一次 ε-greedy 策略改进 =====
                pi = self._policy_improve_at_state(pi=pi, Q=Q, sid=sid, epsilon=eps)

                # 诊断日志（统一命名：SARSA(n=...)/...）
                if step_counter % diag_every == 0:
                    acts = self._allowed(sid)
                    q_s = {aa: Q[sid].get(aa, -np.inf) for aa in acts}
                    self.logger.add_scalar(f"SARSA(n={n})/td_error_abs", abs(float(delta)), ep * 10000 + step_counter)
                    self.logger.add_scalar(f"SARSA(n={n})/q_max_s", max(q_s.values()) if q_s else 0.0,
                                           ep * 10000 + step_counter)
                    self.logger.log(f"[SARSA(n={n})] ep={ep} t={step_counter} |Δ|={abs(float(delta)):.3e}")

                # ===== (4) 环境只“前进一步”：下轮从 (S_1, A_1) 作为新起点 =====
                step_counter += 1
                if not done:
                    # 片段里第 1 步的后继就是新的起点
                    sid = S[1]
                    a = A[1]
                else:
                    break

            # --- 每集收尾：记录回报、ε；可选地同步整张策略的 ε（与现有风格一致） ---
            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            self.logger.add_scalar("SARSA/epsilon", eps, ep)

            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

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

            done = False
            t = 0
            while ((not done) and
                   (t < self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode)):
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
                    self.logger.log(f"[QlearningOnPolicy] ep={ep} t={t} |δ|={abs(float(delta)):.3e}")

                t += 1
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

            done = False
            t = 0
            while ((not done) and
                   (t < self.cfg.max_steps_per_episode if max_steps_per_episode is None else max_steps_per_episode)):
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
                    self.logger.log(f"[QlearningOffPolicy] ep={ep} t={t} |δ|={abs(float(delta)):.3e}")

                t += 1
                # 下一步行动来自“行为策略” π_b
                a_next = sample_action_from_policy(self.rng, pi_b[nsid])
                sid, a = nsid, a_next

            self.logger.add_scalar("episode/return", ep_ret, ep)

        return Q, pi_t