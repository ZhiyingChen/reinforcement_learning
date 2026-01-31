# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from source.grid_world import GridWorld
from source.domain_object import Action, Transition
from source.utils.mdp_ops import (
    q_from_v,
    greedy_policy_from_q,
    expected_backup_under_pi,
    bellman_residual_optimality,
    policy_equal,
)
from source.utils.logger_manager import LoggerManager
from source.utils.timing import record_time_decorator

@dataclass
class PlannerConfig:
    gamma: Optional[float] = None      # 若 None，优先取 env.gamma，否则回退 0.99
    theta: float = 1e-8                # VI/PI 收敛阈值（||V'-V||_inf）
    max_iter: int = 10000              # VI/PI 最大迭代
    eval_theta: float = 1e-8           # 策略评估的阈值（T_pi 迭代）
    eval_max_iter: int = 10000         # 策略评估最大迭代
    tie_breaker: Optional[List[Action]] = None  # 并列动作打破顺序


class VPPlanner:
    """
    统一封装 Value Iteration / Policy Iteration / Truncated Policy Iteration（Modified PI）
    """
    def __init__(self, env: GridWorld, cfg: Optional[PlannerConfig] = None, log_dir="logs/dp", use_tb=True) -> None:
        self.logger = LoggerManager(log_dir, use_tensorboard=use_tb)
        self.logger.log("DPPlanner initialized.")

        self.env = env
        self.P: Dict[int, Dict[Action, List[Transition]]] = env.get_P()
        self.nS = len(self.P)
        self.cfg = cfg or PlannerConfig()
        if self.cfg.tie_breaker is None:
            self.cfg.tie_breaker = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.STAY]

        # 解析 gamma
        if self.cfg.gamma is None:
            self.gamma = getattr(env, "gamma", 0.99)
        else:
            self.gamma = float(self.cfg.gamma)

    # ---------------------------------------------------------------------
    # 公共：策略评估（可完全评估，也可截断 k 次 sweep）
    # ---------------------------------------------------------------------
    def policy_evaluation(
        self,
        pi: Dict[int, Dict[Action, float]],
        V_init: Optional[np.ndarray] = None,
        *,
        theta: Optional[float] = None,
        max_iter: Optional[int] = None,
        sweeps: Optional[int] = None,  # 若指定，则执行固定轮数（截断评估）
    ) -> np.ndarray:
        """
        迭代求解 V^π：
          - 若 sweeps is not None：做 sweeps 次 T_π 备份（Truncated）
          - 否则：直到 ||V'-V||_inf < theta 或达到 max_iter
        """
        V = np.zeros(self.nS, dtype=float) if V_init is None else V_init.copy()
        theta = self.cfg.eval_theta if theta is None else theta
        max_iter = self.cfg.eval_max_iter if max_iter is None else max_iter

        if sweeps is not None:
            # 固定轮数评估
            for _ in range(int(sweeps)):
                V = expected_backup_under_pi(self.P, pi, V, self.gamma)
            return V

        # 直到收敛
        for _ in range(max_iter):
            V_new = expected_backup_under_pi(self.P, pi, V, self.gamma)
            delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            if delta < theta:
                break
        return V

    # ---------------------------------------------------------------------
    # 算法 1：Value Iteration
    # ---------------------------------------------------------------------
    @record_time_decorator('value iteration')
    def value_iteration(
        self,
        V_init: Optional[np.ndarray] = None,
        *,
        theta: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[int, Dict[Action, float]]]:
        V = np.zeros(self.nS, dtype=float) if V_init is None else V_init.copy()
        theta = self.cfg.theta if theta is None else theta
        max_iter = self.cfg.max_iter if max_iter is None else max_iter

        for _ in range(max_iter):
            Q = q_from_v(self.P, V, self.gamma)                            # v_k -> q_k
            pi_greedy = greedy_policy_from_q(Q, self.cfg.tie_breaker)      # q_k -> π_{k+1} (greedy)
            V_new = self.policy_evaluation(pi_greedy, V, sweeps=1)         # v_{k+1} = T_{π_{k+1}} v_k = max_a q_k

            delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            if delta < theta:
                break

        # 再抽取一遍最终贪心策略
        Q = q_from_v(self.P, V, self.gamma)
        pi_star = greedy_policy_from_q(Q, self.cfg.tie_breaker)
        return V, pi_star

    # ---------------------------------------------------------------------
    # 算法 2：Policy Iteration（经典 PI：完整策略评估 + 策略改进）
    # ---------------------------------------------------------------------
    @record_time_decorator('policy iteration')
    def policy_iteration(
        self,
        pi_init: Optional[Dict[int, Dict[Action, float]]] = None,
    ) -> Tuple[np.ndarray, Dict[int, Dict[Action, float]]]:
        # 初始策略：均匀（非终止态）；你也可以传入外部策略
        if pi_init is None:
            pi = self._uniform_policy()
        else:
            pi = pi_init

        while True:
            # 完整评估 V^π
            V = self.policy_evaluation(pi)

            # 策略改进（贪心）
            Q = q_from_v(self.P, V, self.gamma)
            pi_new = greedy_policy_from_q(Q, self.cfg.tie_breaker)

            if policy_equal(pi, pi_new):
                return V, pi_new
            pi = pi_new

    # ---------------------------------------------------------------------
    # 算法 3：Truncated / Modified Policy Iteration
    # 每次只对当前策略做 k 次（或少量）评估 sweep，再策略改进
    # ---------------------------------------------------------------------
    @record_time_decorator('truncated policy iteration')
    def truncated_policy_iteration(
        self,
        *,
        eval_sweeps: int = 5,  # 每轮评估的 sweep 数
        V_init: Optional[np.ndarray] = None,
        pi_init: Optional[Dict[int, Dict[Action, float]]] = None,
        max_outer_iter: int = 1000,
    ) -> Tuple[np.ndarray, Dict[int, Dict[Action, float]]]:
        V = np.zeros(self.nS, dtype=float) if V_init is None else V_init.copy()
        pi = self._uniform_policy() if pi_init is None else pi_init

        for _ in range(max_outer_iter):
            # 截断评估：k 次 T_π 备份
            V = self.policy_evaluation(pi, V, sweeps=eval_sweeps)

            # 策略改进
            Q = q_from_v(self.P, V, self.gamma)
            pi_new = greedy_policy_from_q(Q, self.cfg.tie_breaker)

            if policy_equal(pi, pi_new):
                return V, pi_new
            pi = pi_new

        # 外层达到上限也返回当前近似解
        return V, pi

    # ---------------------------------------------------------------------
    # 辅助：均匀策略
    # ---------------------------------------------------------------------
    def _uniform_policy(self) -> Dict[int, Dict[Action, float]]:
        pi: Dict[int, Dict[Action, float]] = {}
        for s, amap in self.P.items():
            if not amap:
                pi[s] = {Action.STAY: 1.0}
                continue
            acts = list(amap.keys())
            p = 1.0 / len(acts)
            pi[s] = {a: p for a in acts}
        return pi

    # ---------------------------------------------------------------------
    # 评估指标：Bellman 最优性残差
    # ---------------------------------------------------------------------
    def optimality_residual(self, V: np.ndarray) -> float:
        return bellman_residual_optimality(self.P, V, self.gamma)