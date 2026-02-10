# -*- coding: utf-8 -*-
# 路径：source/algorithms/tdfa_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List
import numpy as np
import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from source.utils.logger_manager import LoggerManager
from source.utils.timing import record_time_decorator
from source.utils.policy_ops import sample_action_from_policy  # 与现有 Chapter7 保持一致
from source.domain_object import Action
from source.grid_world import GridWorld

Array = np.ndarray


# ==========================
# 内置：MLP Q 网络（DQN 用）
# ==========================
class QNet(nn.Module):
    """
    简单 MLP：输入 ψ(s)（状态特征），输出每个动作的 Q 向量 Q(s,·)。
    """
    def __init__(self, in_dim: int, n_actions: int, hidden: Tuple[int, ...] = (128,)):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, nA]


# ==========================
# 内置：均匀经验回放（DQN 用）
# ==========================
class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = int(capacity)
        self.buf: List[Tuple[int, int, float, int, bool]] = []
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        if len(self.buf) >= self.capacity:
            # 简单 FIFO
            self.buf.pop(0)
        self.buf.append((int(s), int(a), float(r), int(s2), bool(done)))

    def sample(self, batch_size: int):
        idxs = self.rng.choice(len(self.buf), size=int(batch_size), replace=False)
        S, A, R, S2, D = [], [], [], [], []
        for i in idxs:
            s, a, r, s2, d = self.buf[i]
            S.append(s); A.append(a); R.append(r); S2.append(s2); D.append(d)
        return (np.array(S, dtype=np.int64),
                np.array(A, dtype=np.int64),
                np.array(R, dtype=np.float32),
                np.array(S2, dtype=np.int64),
                np.array(D, dtype=np.uint8))


# ==========================
# TD Function Approx. 配置
# ==========================
@dataclass
class FAConfig:
    gamma: float = 0.99
    seed: int = 42
    log_dir: str = "logs/ch8"
    use_tensorboard: bool = True
    max_steps_per_episode: int = 10_000
    tie_breaker: Optional[List[Action]] = None  # 并列破除顺序


# ==========================
# 主类：TDFAPlanner（Chapter 8）
# ==========================
class TDFAPlanner:
    r"""
    Chapter 8：Value Function Approximation（控制问题）
      - SARSA with linear FA
      - Q-learning with linear FA (on-policy 版本)
      - Deep Q-learning（on / off-policy）

    说明：
      * 线性 FA：  \hat{q}(s,a; w) = <w, φ(s,a)>
        SARSA-FA: w <- w + α * [r + γ * \hat{q}(s',a';w) - \hat{q}(s,a;w)] * φ(s,a)
        QL-FA(on): w <- w + α * [r + γ * max_{a'} \hat{q}(s',a';w) - \hat{q}(s,a;w)] * φ(s,a)
        （与 Lecture 8 一致）                                          # 参见讲义 P47-53
      * DQN： 主网络 w / 目标网络 w_T；均匀经验回放；Huber Loss；定期同步目标网络  # 参见讲义 P55-61, P65
    """

    def __init__(self, env: GridWorld, cfg: Optional[FAConfig] = None) -> None:
        self.env = env
        self.cfg = cfg or FAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.logger = LoggerManager(self.cfg.log_dir, self.cfg.use_tensorboard)
        self.gamma = float(getattr(env, "gamma", self.cfg.gamma))
        self.nS = len(env.id2s)
        # 与 Chapter 7 保持一致的 tie-break 顺序
        self.tie = self.cfg.tie_breaker or [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.STAY]

    # -----------------------
    # 公共工具
    # -----------------------
    def _allowed(self, sid: int) -> List[Action]:
        return self.env.allowed_actions(self.env.id2s[sid])

    # ========== 线性特征 ==========
    @staticmethod
    def _state_features(env: GridWorld, sid: int) -> np.ndarray:
        """
        ψ(s)：状态特征（可按需替换/扩展）。
        这里使用 6 维多项式基：[1, x, y, x^2, y^2, x*y]，x,y ∈ [-1,1]（行/列归一化）。
        """
        r, c = env.id2s[sid]
        x = 2.0 * r / max(1, (env.h - 1)) - 1.0
        y = 2.0 * c / max(1, (env.w - 1)) - 1.0
        return np.array([1.0, x, y, x * x, y * y, x * y], dtype=np.float32)

    @staticmethod
    def _sa_features(env: GridWorld, sid: int, a: Action, use_interaction: bool = True) -> np.ndarray:
        """
        φ(s,a)：将 ψ(s) 与 动作 one-hot（以及可选交互项）拼接得到。
        交互项能增强表达能力，但也会增大维度。
        """
        psi = TDFAPlanner._state_features(env, sid)                 # dψ
        nA = len(Action.all())
        one = np.zeros(nA, dtype=np.float32)                        # dA
        one[a.value] = 1.0
        if use_interaction:
            inter = np.zeros((nA, psi.shape[0]), dtype=np.float32)  # nA × dψ
            inter[a.value] = psi
            return np.concatenate([psi, one, inter.ravel()], dtype=np.float32)
        return np.concatenate([psi, one], dtype=np.float32)

    def _epsilon_greedy_from_qhat_linear(self, sid: int, w: np.ndarray, epsilon: float, use_interaction: bool=True) -> Dict[Action, float]:
        r"""
        基于线性近似 \hat{q}(s,a;w) 的 ε-greedy 分布。
        """
        acts = self._allowed(sid)
        q_vals = []
        for a in acts:
            phi = self._sa_features(self.env, sid, a, use_interaction=use_interaction)
            q_vals.append(float(np.dot(w, phi)))
        max_q = max(q_vals) if q_vals else -np.inf
        best = [a for a, q in zip(acts, q_vals) if math.isclose(q, max_q, rel_tol=1e-9, abs_tol=1e-12)]
        a_star = sorted(best, key=lambda a: self.tie.index(a))[0] if best else Action.STAY
        nA = len(acts)
        base = epsilon / nA
        pi_s = {a: base for a in acts}
        pi_s[a_star] = 1.0 - epsilon + base
        return pi_s

    def _greedy_pi_from_qhat_linear(self, w: np.ndarray, use_interaction: bool=True) -> Dict[int, Dict[Action, float]]:
        """
        输出 one-hot 的贪心策略（用于渲染）。
        """
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed(sid)
            if not acts:
                pi[sid] = {Action.STAY: 1.0}
                continue
            q_s = {a: float(np.dot(w, self._sa_features(self.env, sid, a, use_interaction=use_interaction))) for a in acts}
            max_q = max(q_s.values())
            best = [a for a, q in q_s.items() if math.isclose(q, max_q, rel_tol=1e-9, abs_tol=1e-12)]
            a_star = sorted(best, key=lambda a: self.tie.index(a))[0]
            pi[sid] = {a: (1.0 if a == a_star else 0.0) for a in Action}
        return pi

    # ========================================================
    # 1) SARSA with Linear Function Approximation (on-policy)
    # ========================================================
    @record_time_decorator("SARSA-Linear")
    def sarsa_linear(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.05,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        min_epsilon: float = 0.01,
        w_init: Optional[np.ndarray] = None,
        use_interaction: bool = True,
        diag_every: int = 200,
    ) -> Tuple[np.ndarray, Dict[int, Dict[Action, float]]]:
        r"""
        线性 SARSA-FA（on-policy）：
            w <- w + α * δ * φ(s,a)
            δ = r + γ * \hat{q}(s',a';w) - \hat{q}(s,a;w)
        参照 Lecture 8（P47-49）的伪代码/公式。  # 引用
        """
        # 初始化
        phi_dim = self._sa_features(self.env, 0, Action.UP, use_interaction=use_interaction).shape[0]
        w = np.zeros(phi_dim, dtype=np.float32) if w_init is None else w_init.astype(np.float32)

        eps = float(epsilon)
        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            # 构造当前 ε-greedy 行为/目标策略（on-policy）
            pi = {s: self._epsilon_greedy_from_qhat_linear(s, w, eps, use_interaction=use_interaction) for s in range(self.nS)}
            a = sample_action_from_policy(self.rng, pi[sid])
            ep_ret = 0.0

            t = 0
            done = False
            while t < self.cfg.max_steps_per_episode and not done:
                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)

                # TD 目标与增量
                phi_sa = self._sa_features(self.env, sid, a, use_interaction=use_interaction)
                q_sa = float(np.dot(w, phi_sa))
                if done:
                    target = float(r)
                else:
                    a_next = sample_action_from_policy(self.rng, pi[nsid])
                    phi_next = self._sa_features(self.env, nsid, a_next, use_interaction=use_interaction)
                    target = float(r) + self.gamma * float(np.dot(w, phi_next))
                delta = target - q_sa
                w += alpha * delta * phi_sa

                # 仅在 s_t 局部做一次 ε-greedy 政策改进
                pi[sid] = self._epsilon_greedy_from_qhat_linear(sid, w, eps, use_interaction=use_interaction)

                # 诊断日志
                if t % diag_every == 0:
                    self.logger.add_scalar("SARSA-Linear/td_error_abs", abs(float(delta)), ep * 10_000 + t)

                t += 1
                sid, a = nsid, a_next

            # 每集收尾
            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            self.logger.add_scalar("SARSA/epsilon", eps, ep)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

        # 最终贪心策略（用于可视化）
        pi_star = self._greedy_pi_from_qhat_linear(w, use_interaction=use_interaction)
        return w, pi_star

    # ============================================================
    # 2) Q-learning with Linear Function Approx. (on-policy 版本)
    # ============================================================
    @record_time_decorator("Q-Learning-Linear-On")
    def q_learning_linear_on(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.05,
        epsilon: float = 0.1,
        epsilon_decay: Optional[float] = None,
        min_epsilon: float = 0.01,
        w_init: Optional[np.ndarray] = None,
        use_interaction: bool = True,
        diag_every: int = 200,
    ) -> Tuple[np.ndarray, Dict[int, Dict[Action, float]]]:
        r"""
        线性 Q-learning（on-policy 版本）：
            w <- w + α * δ * φ(s,a)
            δ = r + γ * max_{a'} \hat{q}(s',a';w) - \hat{q}(s,a;w)
        采样用 ε-greedy(w)，目标项为 max-backup（Lecture 8 P51-53）。  # 引用
        """
        phi_dim = self._sa_features(self.env, 0, Action.UP, use_interaction=use_interaction).shape[0]
        w = np.zeros(phi_dim, dtype=np.float32) if w_init is None else w_init.astype(np.float32)
        eps = float(epsilon)

        def qmax_next(nsid: int) -> float:
            acts = self._allowed(nsid)
            if not acts:
                return 0.0
            vals = [float(np.dot(w, self._sa_features(self.env, nsid, a, use_interaction=use_interaction))) for a in acts]
            return max(vals)

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            # on-policy 行为策略
            pi = {s: self._epsilon_greedy_from_qhat_linear(s, w, eps, use_interaction=use_interaction) for s in range(self.nS)}
            a = sample_action_from_policy(self.rng, pi[sid])
            ep_ret = 0.0

            t = 0
            done = False
            while t < self.cfg.max_steps_per_episode and not done:
                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)

                phi_sa = self._sa_features(self.env, sid, a, use_interaction=use_interaction)
                q_sa = float(np.dot(w, phi_sa))
                if done:
                    target = float(r)
                else:
                    target = float(r) + self.gamma * qmax_next(nsid)
                delta = target - q_sa
                w += alpha * delta * phi_sa

                # 局部策略改进（保持 on-policy）
                pi[sid] = self._epsilon_greedy_from_qhat_linear(sid, w, eps, use_interaction=use_interaction)

                if t % diag_every == 0:
                    self.logger.add_scalar("Q-Linear-On/td_error_abs", abs(float(delta)), ep * 10_000 + t)

                t += 1
                # 下一步 a 来自“更新前”的策略 π_t(s')
                a_next = sample_action_from_policy(self.rng, pi[nsid])
                sid, a = nsid, a_next

            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            self.logger.add_scalar("Q-Linear/epsilon", eps, ep)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

        pi_star = self._greedy_pi_from_qhat_linear(w, use_interaction=use_interaction)
        return w, pi_star

    # ======================================
    # 3) DQN（on-policy；均匀经验回放 + 目标网）
    # ======================================
    @record_time_decorator("DQN-OnPolicy")
    def dqn_on_policy(
        self,
        *,
        num_episodes: int = 1000,
        epsilon: float = 0.2,
        epsilon_decay: Optional[float] = 0.995,
        min_epsilon: float = 0.05,
        lr: float = 1e-3,
        hidden: Tuple[int, ...] = (128,),
        batch_size: int = 64,
        buffer_size: int = 50_000,
        warmup: int = 1_000,
        target_sync_every: int = 500,  # 按“步”同步目标网
        diag_every: int = 200,
    ) -> Tuple[QNet, QNet, Dict[int, Dict[Action, float]]]:
        """
        DQN（on-policy）：
          - 收集数据时用当前主网的 ε-greedy
          - 均匀经验回放 + HuberLoss
          - 定期同步目标网络参数
        参照 Lecture 8（P55-61）思路。
        """
        obs_dim = self._state_features(self.env, 0).shape[0]
        nA = len(Action.all())
        q = QNet(obs_dim, nA, hidden)
        tq = QNet(obs_dim, nA, hidden)
        tq.load_state_dict(q.state_dict())
        opt = optim.Adam(q.parameters(), lr=lr)
        huber = nn.SmoothL1Loss()
        buf = ReplayBuffer(buffer_size, seed=self.cfg.seed)

        def q_values(net: QNet, sids: Array) -> np.ndarray:
            X = np.stack([self._state_features(self.env, int(s)) for s in sids])
            with torch.no_grad():
                out = net(torch.from_numpy(X).float())
                return out.cpu().numpy()

        eps = float(epsilon)
        global_step = 0

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            ep_ret = 0.0

            for t in range(self.cfg.max_steps_per_episode):
                # ε-greedy（on-policy 采样）
                if self.rng.random() < eps:
                    a = self.rng.choice(self._allowed(sid))
                else:
                    q_s = q_values(q, np.array([sid], dtype=np.int64))[0]  # [nA]
                    # 仅挑选允许动作中的最大值
                    a = sorted(self._allowed(sid), key=lambda A: q_s[A.value], reverse=True)[0]

                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)
                buf.push(sid, a.value, float(r), nsid, bool(done))
                sid = nsid
                global_step += 1

                # 训练
                if len(buf) >= max(warmup, batch_size):
                    S, A, R, S2, D = buf.sample(batch_size)
                    # 目标：y = r + γ * max_a' Q_T(s',a')
                    with torch.no_grad():
                        q_next = q_values(tq, S2)                        # [B, nA]
                        # 仅考虑允许动作的最大值：为简单，假设全部动作可用（GridWorld 里终止态仅 STAY）
                        max_next = torch.from_numpy(q_next).float().max(dim=1).values
                        y = torch.from_numpy(R).float() + self.gamma * (1.0 - torch.from_numpy(D).float()) * max_next

                    # 当前 Q(s,a)
                    X = np.stack([self._state_features(self.env, int(s)) for s in S])
                    q_sa_all = q(torch.from_numpy(X).float())            # [B, nA]
                    q_sa = q_sa_all.gather(1, torch.from_numpy(A).long().view(-1, 1)).squeeze(1)

                    loss = huber(q_sa, y)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), max_norm=10.0)
                    opt.step()

                    if global_step % diag_every == 0:
                        self.logger.add_scalar("DQN/loss", float(loss.item()), global_step)

                # 同步目标网
                if global_step % target_sync_every == 0:
                    tq.load_state_dict(q.state_dict())

                if done:
                    break

            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            if epsilon_decay is not None:
                eps = max(min_epsilon, eps * epsilon_decay)

        # 最终贪心策略
        pi_star: Dict[int, Dict[Action, float]] = {}
        for s in range(self.nS):
            q_s = q_values(q, np.array([s], dtype=np.int64))[0]
            a_star = sorted(self._allowed(s), key=lambda A: q_s[A.value], reverse=True)[0]
            pi_star[s] = {a: (1.0 if a == a_star else 0.0) for a in Action}
        return q, tq, pi_star

    # =======================================
    # 4) DQN（off-policy；行为/目标策略分离）
    # =======================================
    @record_time_decorator("DQN-OffPolicy")
    def dqn_off_policy(
        self,
        *,
        num_episodes: int = 1000,
        behavior_epsilon: float = 0.3,   # 行为策略 ε（用于采样）
        target_epsilon: float = 0.05,    # 目标策略 ε（仅用于导出策略/评估，可设为小 ε 或贪心）
        lr: float = 1e-3,
        hidden: Tuple[int, ...] = (128,),
        batch_size: int = 64,
        buffer_size: int = 50_000,
        warmup: int = 1_000,
        target_sync_every: int = 500,
        diag_every: int = 200,
    ) -> Tuple[QNet, QNet, Dict[int, Dict[Action, float]]]:
        """
        DQN（off-policy）：
          - 采样用行为策略（较大 ε）
          - 训练主网并用目标网生成 y = r + γ max_{a'} Q_T(s',a')
          - 均匀回放 + HuberLoss + 目标网同步
        参照 Lecture 8 伪代码（P65）。                                        # 引用
        """
        obs_dim = self._state_features(self.env, 0).shape[0]
        nA = len(Action.all())
        q = QNet(obs_dim, nA, hidden)
        tq = QNet(obs_dim, nA, hidden)
        tq.load_state_dict(q.state_dict())
        opt = optim.Adam(q.parameters(), lr=lr)
        huber = nn.SmoothL1Loss()
        buf = ReplayBuffer(buffer_size, seed=self.cfg.seed)

        def q_values(net: QNet, sids: Array) -> np.ndarray:
            X = np.stack([self._state_features(self.env, int(s)) for s in sids])
            with torch.no_grad():
                out = net(torch.from_numpy(X).float())
                return out.cpu().numpy()

        global_step = 0

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            ep_ret = 0.0

            for t in range(self.cfg.max_steps_per_episode):
                # 行为策略 ε_b（off-policy 采样）
                if self.rng.random() < behavior_epsilon:
                    a = self.rng.choice(self._allowed(sid))
                else:
                    q_s = q_values(q, np.array([sid], dtype=np.int64))[0]
                    a = sorted(self._allowed(sid), key=lambda A: q_s[A.value], reverse=True)[0]

                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)
                buf.push(sid, a.value, float(r), nsid, bool(done))
                sid = nsid
                global_step += 1

                # 训练
                if len(buf) >= max(warmup, batch_size):
                    S, A, R, S2, D = buf.sample(batch_size)
                    with torch.no_grad():
                        q_next = q_values(tq, S2)
                        max_next = torch.from_numpy(q_next).float().max(dim=1).values
                        y = torch.from_numpy(R).float() + self.gamma * (1.0 - torch.from_numpy(D).float()) * max_next
                    X = np.stack([self._state_features(self.env, int(s)) for s in S])
                    q_sa_all = q(torch.from_numpy(X).float())
                    q_sa = q_sa_all.gather(1, torch.from_numpy(A).long().view(-1, 1)).squeeze(1)
                    loss = huber(q_sa, y)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), max_norm=10.0)
                    opt.step()

                    if global_step % diag_every == 0:
                        self.logger.add_scalar("DQN-Off/loss", float(loss.item()), global_step)

                if global_step % target_sync_every == 0:
                    tq.load_state_dict(q.state_dict())

                if done:
                    break

            self.logger.add_scalar("episode/return", float(ep_ret), ep)

        # 用较小 ε（或贪心）导出目标策略
        pi_star: Dict[int, Dict[Action, float]] = {}
        for s in range(self.nS):
            q_s = q_values(q, np.array([s], dtype=np.int64))[0]
            a_star = sorted(self._allowed(s), key=lambda A: q_s[A.value], reverse=True)[0]
            # 这里导出 one-hot（若你想查看 ε_t 的行为策略，可自行构造分布）
            pi_star[s] = {a: (1.0 if a == a_star else 0.0) for a in Action}
        return q, tq, pi_star