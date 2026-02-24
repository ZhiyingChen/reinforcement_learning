# -*- coding: utf-8 -*-
# 路径：source/algorithms/pgac_planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from source.grid_world import GridWorld
from source.domain_object import Action
from source.utils.logger_manager import LoggerManager
from source.utils.timing import record_time_decorator
from source.utils.policy_ops import sample_action_from_policy

Array = np.ndarray


@dataclass
class PGACConfig:
    """Policy Gradient / Actor-Critic 配置

    - 与现有 Chapter7/8 Planner 结构风格保持一致（env.step 在线更新、LoggerManager、record_time_decorator）
    - 针对 GridWorld（离散动作）实现 Lecture 9 & 10：REINFORCE, QAC, A2C, Off-policy AC, DPG
    """
    gamma: Optional[float] = None
    seed: int = 42
    log_dir: str = "logs/ch9_10"
    use_tensorboard: bool = True
    max_steps_per_episode: int = 10_000
    tie_breaker: Optional[List[Action]] = None


class PGACPlanner:
    def __init__(self, env: GridWorld, cfg: Optional[PGACConfig] = None):
        self.env = env
        self.cfg = cfg or PGACConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.logger = LoggerManager(self.cfg.log_dir, self.cfg.use_tensorboard)

        self.gamma = float(getattr(env, "gamma", 0.99) if self.cfg.gamma is None else self.cfg.gamma)
        self.nS = len(env.id2s)
        self.nA = len(Action.all())
        self.tie = self.cfg.tie_breaker or [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.STAY]

        self._step_idx = 0
        self.logger.log("PGACPlanner initialized.")

    # ============================================================
    # Features（保持自包含；风格参考你 Chapter8 TDFAPlanner 的多项式基）
    # ============================================================
    @staticmethod
    def state_features(env: GridWorld, sid: int) -> Array:
        """ψ(s) = [1, x, y, x^2, y^2, x*y], x,y ∈ [-1,1]"""
        r, c = env.id2s[sid]
        x = 2.0 * r / max(1, (env.h - 1)) - 1.0
        y = 2.0 * c / max(1, (env.w - 1)) - 1.0
        return np.array([1.0, x, y, x * x, y * y, x * y], dtype=np.float32)

    @staticmethod
    def sa_features(env: GridWorld, sid: int, a: Action, use_interaction: bool = True) -> Array:
        """φ(s,a) = concat(ψ(s), onehot(a), optional interaction(onehot(a) ⊗ ψ(s)))"""
        psi = PGACPlanner.state_features(env, sid)
        nA = len(Action.all())
        one = np.zeros(nA, dtype=np.float32)
        one[a.value] = 1.0
        if use_interaction:
            inter = np.zeros((nA, psi.shape[0]), dtype=np.float32)
            inter[a.value] = psi
            return np.concatenate([psi, one, inter.ravel()], dtype=np.float32)
        return np.concatenate([psi, one], dtype=np.float32)

    def _allowed(self, sid: int) -> List[Action]:
        return self.env.allowed_actions(self.env.id2s[sid])

    # ============================================================
    # Stochastic policy: softmax over linear preferences
    # π(a|s;θ) ∝ exp(<θ, φ(s,a)>)
    # ============================================================
    def _policy_probs(self, sid: int, theta: Array, *, use_interaction: bool = True) -> Dict[Action, float]:
        acts = self._allowed(sid)
        if not acts:
            return {Action.STAY: 1.0}
        prefs = []
        for a in acts:
            phi = self.sa_features(self.env, sid, a, use_interaction=use_interaction)
            prefs.append(float(np.dot(theta, phi)))
        z = np.array(prefs, dtype=np.float64)
        z = z - np.max(z)
        exp = np.exp(z)
        p = exp / np.sum(exp)
        return {a: float(pi) for a, pi in zip(acts, p)}

    def _grad_log_pi(self, sid: int, a_taken: Action, theta: Array, *, use_interaction: bool = True) -> Array:
        """∇θ log π(a|s;θ) = φ(s,a) - E_{a'~π}[φ(s,a')]"""
        pi = self._policy_probs(sid, theta, use_interaction=use_interaction)
        acts = list(pi.keys())
        feat_dim = self.sa_features(self.env, sid, acts[0], use_interaction=use_interaction).shape[0]
        mean_phi = np.zeros(feat_dim, dtype=np.float32)
        for a, p in pi.items():
            mean_phi += float(p) * self.sa_features(self.env, sid, a, use_interaction=use_interaction)
        phi_taken = self.sa_features(self.env, sid, a_taken, use_interaction=use_interaction)
        return (phi_taken - mean_phi).astype(np.float32)

    def _greedy_policy_from_theta(self, theta: Array, *, use_interaction: bool = True) -> Dict[int, Dict[Action, float]]:
        """导出 one-hot 贪心策略（用于 render）"""
        pi: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            acts = self._allowed(sid)
            if not acts:
                pi[sid] = {Action.STAY: 1.0}
                continue
            scores = {a: float(np.dot(theta, self.sa_features(self.env, sid, a, use_interaction=use_interaction)))
                      for a in acts}
            best_val = max(scores.values())
            best = [a for a, v in scores.items() if np.isclose(v, best_val)]
            a_star = sorted(best, key=lambda aa: self.tie.index(aa))[0]
            pi[sid] = {a: (1.0 if a == a_star else 0.0) for a in Action}
        return pi

    # ============================================================
    # REINFORCE helpers
    # ============================================================
    def _rollout(self, theta: Array, *, use_interaction: bool = True, max_steps: Optional[int] = None):
        max_steps = self.cfg.max_steps_per_episode if max_steps is None else int(max_steps)
        self.env.reset()
        sid = self.env.sid(self.env.s)
        S: List[int] = []
        A: List[Action] = []
        R: List[float] = []
        for _t in range(max_steps):
            pi = self._policy_probs(sid, theta, use_interaction=use_interaction)
            a = sample_action_from_policy(self.rng, pi)
            nsid, r, done, _ = self.env.step(a)
            S.append(sid)
            A.append(a)
            R.append(float(r))
            sid = nsid
            if done:
                break
        return S, A, R

    def _returns(self, rewards: List[float]) -> List[float]:
        G = 0.0
        out = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            out[t] = float(G)
        return out

    # ============================================================
    # Lecture 9: REINFORCE
    # ============================================================
    @record_time_decorator("REINFORCE")
    def reinforce(
        self,
        *,
        num_episodes: int = 2000,
        alpha: float = 0.02,
        use_interaction: bool = True,
        max_steps_per_episode: Optional[int] = None,
        normalize_return: bool = False,
        diag_every: int = 50,
        theta_init: Optional[Array] = None,
    ) -> Tuple[Array, Dict[int, Dict[Action, float]]]:
        feat_dim = self.sa_features(self.env, 0, Action.UP, use_interaction=use_interaction).shape[0]
        theta = np.zeros(feat_dim, dtype=np.float32) if theta_init is None else theta_init.astype(np.float32)

        for ep in range(num_episodes):
            S, A, R = self._rollout(theta, use_interaction=use_interaction, max_steps=max_steps_per_episode)
            Gs = self._returns(R)

            if normalize_return and len(Gs) > 1:
                g = np.array(Gs, dtype=np.float32)
                g = (g - g.mean()) / (g.std() + 1e-8)
                Gs = [float(x) for x in g]

            ep_ret = float(sum(R))
            self.logger.add_scalar("episode/return", ep_ret, ep)

            for sid, a, Gt in zip(S, A, Gs):
                grad = self._grad_log_pi(sid, a, theta, use_interaction=use_interaction)
                theta += float(alpha) * float(Gt) * grad
                self._step_idx += 1

            if ep % diag_every == 0:
                self.logger.log(f"[REINFORCE] ep={ep} len={len(S)} return={ep_ret:.3f}")

        pi_star = self._greedy_policy_from_theta(theta, use_interaction=use_interaction)
        return theta, pi_star

    # ============================================================
    # Lecture 10: Q Actor-Critic (QAC)
    # ============================================================
    @record_time_decorator("QAC")
    def q_actor_critic(
        self,
        *,
        num_episodes: int = 2000,
        alpha_theta: float = 0.01,
        alpha_w: float = 0.05,
        use_interaction_actor: bool = True,
        use_interaction_critic: bool = True,
        max_steps_per_episode: Optional[int] = None,
        diag_every: int = 200,
        theta_init: Optional[Array] = None,
        w_init: Optional[Array] = None,
    ) -> Tuple[Array, Array, Dict[int, Dict[Action, float]]]:
        feat_pi = self.sa_features(self.env, 0, Action.UP, use_interaction=use_interaction_actor).shape[0]
        theta = np.zeros(feat_pi, dtype=np.float32) if theta_init is None else theta_init.astype(np.float32)

        feat_q = self.sa_features(self.env, 0, Action.UP, use_interaction=use_interaction_critic).shape[0]
        w = np.zeros(feat_q, dtype=np.float32) if w_init is None else w_init.astype(np.float32)

        max_steps = self.cfg.max_steps_per_episode if max_steps_per_episode is None else int(max_steps_per_episode)

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            a = sample_action_from_policy(self.rng, self._policy_probs(sid, theta, use_interaction=use_interaction_actor))

            ep_ret = 0.0
            done = False
            t = 0
            while (not done) and (t < max_steps):
                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)

                if not done:
                    a_next = sample_action_from_policy(
                        self.rng, self._policy_probs(nsid, theta, use_interaction=use_interaction_actor)
                    )
                else:
                    a_next = Action.STAY

                phi_sa = self.sa_features(self.env, sid, a, use_interaction=use_interaction_critic)
                q_sa = float(np.dot(w, phi_sa))
                if done:
                    target = float(r)
                else:
                    phi_next = self.sa_features(self.env, nsid, a_next, use_interaction=use_interaction_critic)
                    target = float(r) + self.gamma * float(np.dot(w, phi_next))
                delta = target - q_sa
                w += float(alpha_w) * float(delta) * phi_sa

                # actor uses updated critic
                q_for_actor = float(np.dot(w, phi_sa))
                grad_log = self._grad_log_pi(sid, a, theta, use_interaction=use_interaction_actor)
                theta += float(alpha_theta) * q_for_actor * grad_log

                if (self._step_idx % diag_every) == 0:
                    self.logger.add_scalar("QAC/td_error_abs", abs(float(delta)), self._step_idx)
                self._step_idx += 1

                sid, a = nsid, a_next
                t += 1

            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            if ep % max(1, diag_every // 5) == 0:
                self.logger.log(f"[QAC] ep={ep} return={ep_ret:.3f} steps={t}")

        pi_star = self._greedy_policy_from_theta(theta, use_interaction=use_interaction_actor)
        return theta, w, pi_star

    # ============================================================
    # Lecture 10: Advantage Actor-Critic (A2C)
    # ============================================================
    @record_time_decorator("A2C")
    def a2c(
        self,
        *,
        num_episodes: int = 2000,
        alpha_theta: float = 0.02,
        alpha_w: float = 0.05,
        use_interaction_actor: bool = True,
        max_steps_per_episode: Optional[int] = None,
        diag_every: int = 200,
        theta_init: Optional[Array] = None,
        w_init: Optional[Array] = None,
    ) -> Tuple[Array, Array, Dict[int, Dict[Action, float]]]:
        feat_pi = self.sa_features(self.env, 0, Action.UP, use_interaction=use_interaction_actor).shape[0]
        theta = np.zeros(feat_pi, dtype=np.float32) if theta_init is None else theta_init.astype(np.float32)

        feat_v = self.state_features(self.env, 0).shape[0]
        w = np.zeros(feat_v, dtype=np.float32) if w_init is None else w_init.astype(np.float32)

        max_steps = self.cfg.max_steps_per_episode if max_steps_per_episode is None else int(max_steps_per_episode)

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            ep_ret = 0.0
            done = False
            t = 0

            while (not done) and (t < max_steps):
                a = sample_action_from_policy(
                    self.rng, self._policy_probs(sid, theta, use_interaction=use_interaction_actor)
                )
                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)

                psi_s = self.state_features(self.env, sid)
                v_s = float(np.dot(w, psi_s))
                v_next = 0.0 if done else float(np.dot(w, self.state_features(self.env, nsid)))

                delta = float(r) + self.gamma * v_next - v_s
                w += float(alpha_w) * delta * psi_s

                grad_log = self._grad_log_pi(sid, a, theta, use_interaction=use_interaction_actor)
                theta += float(alpha_theta) * delta * grad_log

                if (self._step_idx % diag_every) == 0:
                    self.logger.add_scalar("A2C/td_error_abs", abs(float(delta)), self._step_idx)
                self._step_idx += 1

                sid = nsid
                t += 1

            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            if ep % max(1, diag_every // 5) == 0:
                self.logger.log(f"[A2C] ep={ep} return={ep_ret:.3f} steps={t}")

        pi_star = self._greedy_policy_from_theta(theta, use_interaction=use_interaction_actor)
        return theta, w, pi_star

    # ============================================================
    # Lecture 10: Off-policy Actor-Critic (importance sampling)
    # ============================================================
    @record_time_decorator("OffPolicy-AC")
    def off_policy_actor_critic(
        self,
        *,
        num_episodes: int = 2000,
        alpha_theta: float = 0.02,
        alpha_w: float = 0.05,
        behavior_epsilon: float = 0.3,
        use_interaction_actor: bool = True,
        max_steps_per_episode: Optional[int] = None,
        rho_clip: Optional[float] = None,
        diag_every: int = 200,
        theta_init: Optional[Array] = None,
        w_init: Optional[Array] = None,
    ) -> Tuple[Array, Array, Dict[int, Dict[Action, float]]]:
        feat_pi = self.sa_features(self.env, 0, Action.UP, use_interaction=use_interaction_actor).shape[0]
        theta = np.zeros(feat_pi, dtype=np.float32) if theta_init is None else theta_init.astype(np.float32)

        feat_v = self.state_features(self.env, 0).shape[0]
        w = np.zeros(feat_v, dtype=np.float32) if w_init is None else w_init.astype(np.float32)

        max_steps = self.cfg.max_steps_per_episode if max_steps_per_episode is None else int(max_steps_per_episode)

        def behavior_policy(sid: int) -> Dict[Action, float]:
            pi = self._policy_probs(sid, theta, use_interaction=use_interaction_actor)
            acts = list(pi.keys())
            uni = 1.0 / len(acts)
            mu = {a: (1.0 - behavior_epsilon) * pi[a] + behavior_epsilon * uni for a in acts}
            tot = sum(mu.values())
            return {a: float(p / tot) for a, p in mu.items()}

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            ep_ret = 0.0
            done = False
            t = 0

            while (not done) and (t < max_steps):
                mu_s = behavior_policy(sid)
                a = sample_action_from_policy(self.rng, mu_s)
                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)

                pi_s = self._policy_probs(sid, theta, use_interaction=use_interaction_actor)
                rho = float(pi_s.get(a, 0.0)) / max(1e-12, float(mu_s.get(a, 1e-12)))
                if rho_clip is not None:
                    rho = float(min(rho_clip, rho))

                psi_s = self.state_features(self.env, sid)
                v_s = float(np.dot(w, psi_s))
                v_next = 0.0 if done else float(np.dot(w, self.state_features(self.env, nsid)))
                delta = float(r) + self.gamma * v_next - v_s

                w += float(alpha_w) * rho * delta * psi_s
                grad_log = self._grad_log_pi(sid, a, theta, use_interaction=use_interaction_actor)
                theta += float(alpha_theta) * rho * delta * grad_log

                if (self._step_idx % diag_every) == 0:
                    self.logger.add_scalar("OffAC/rho", float(rho), self._step_idx)
                    self.logger.add_scalar("OffAC/td_error_abs", abs(float(delta)), self._step_idx)
                self._step_idx += 1

                sid = nsid
                t += 1

            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            if ep % max(1, diag_every // 5) == 0:
                self.logger.log(f"[OffAC] ep={ep} return={ep_ret:.3f} steps={t}")

        pi_star = self._greedy_policy_from_theta(theta, use_interaction=use_interaction_actor)
        return theta, w, pi_star

    # ============================================================
    # Lecture 10: DPG (discrete differentiable surrogate)
    # ============================================================
    @record_time_decorator("DPG")
    def dpg(
        self,
        *,
        num_episodes: int = 2000,
        alpha_theta: float = 0.02,
        alpha_w: float = 0.05,
        behavior_epsilon: float = 0.3,
        max_steps_per_episode: Optional[int] = None,
        use_interaction_critic: bool = True,
        diag_every: int = 200,
        theta_init: Optional[Array] = None,
        w_init: Optional[Array] = None,
    ) -> Tuple[Array, Array, Dict[int, Dict[Action, float]]]:
        feat_s = self.state_features(self.env, 0).shape[0]
        theta = np.zeros((self.nA, feat_s), dtype=np.float32) if theta_init is None else theta_init.astype(np.float32)

        feat_q = self.sa_features(self.env, 0, Action.UP, use_interaction=use_interaction_critic).shape[0]
        w = np.zeros(feat_q, dtype=np.float32) if w_init is None else w_init.astype(np.float32)

        max_steps = self.cfg.max_steps_per_episode if max_steps_per_episode is None else int(max_steps_per_episode)

        def mu_probs(sid: int) -> np.ndarray:
            psi = self.state_features(self.env, sid)
            z = theta @ psi
            z = z - np.max(z)
            exp = np.exp(z.astype(np.float64))
            p = exp / np.sum(exp)
            return p.astype(np.float32)

        def q_hat_all(sid: int) -> np.ndarray:
            out = np.zeros(self.nA, dtype=np.float32)
            for a in Action.all():
                phi = self.sa_features(self.env, sid, a, use_interaction=use_interaction_critic)
                out[a.value] = float(np.dot(w, phi))
            return out

        for ep in range(num_episodes):
            self.env.reset()
            sid = self.env.sid(self.env.s)
            ep_ret = 0.0
            done = False
            t = 0

            while (not done) and (t < max_steps):
                p = mu_probs(sid)
                if self.rng.random() < behavior_epsilon:
                    a = self.rng.choice(self._allowed(sid))
                else:
                    acts = self._allowed(sid)
                    a = sorted(acts, key=lambda aa: p[aa.value], reverse=True)[0]

                nsid, r, done, _ = self.env.step(a)
                ep_ret += float(r)

                phi_sa = self.sa_features(self.env, sid, a, use_interaction=use_interaction_critic)
                q_sa = float(np.dot(w, phi_sa))

                if done:
                    target = float(r)
                else:
                    p_next = mu_probs(nsid)
                    q_next_all = q_hat_all(nsid)
                    q_next_mu = float(np.dot(p_next, q_next_all))  # q(s', μ(s'))
                    target = float(r) + self.gamma * q_next_mu

                delta = target - q_sa
                w += float(alpha_w) * float(delta) * phi_sa

                # actor: maximize μ(s)·q(s,·)
                psi_s = self.state_features(self.env, sid)
                q_vec = q_hat_all(sid)
                p = mu_probs(sid)
                q_bar = float(np.dot(p, q_vec))
                grad_z = p * (q_vec - q_bar)
                theta += float(alpha_theta) * grad_z.reshape(-1, 1) * psi_s.reshape(1, -1)

                if (self._step_idx % diag_every) == 0:
                    self.logger.add_scalar("DPG/td_error_abs", abs(float(delta)), self._step_idx)
                self._step_idx += 1

                sid = nsid
                t += 1

            self.logger.add_scalar("episode/return", float(ep_ret), ep)
            if ep % max(1, diag_every // 5) == 0:
                self.logger.log(f"[DPG] ep={ep} return={ep_ret:.3f} steps={t}")

        pi_star: Dict[int, Dict[Action, float]] = {}
        for sid in range(self.nS):
            p = mu_probs(sid)
            acts = self._allowed(sid)
            if not acts:
                pi_star[sid] = {Action.STAY: 1.0}
                continue
            a_star = sorted(acts, key=lambda aa: p[aa.value], reverse=True)[0]
            pi_star[sid] = {a: (1.0 if a == a_star else 0.0) for a in Action}

        return theta, w, pi_star