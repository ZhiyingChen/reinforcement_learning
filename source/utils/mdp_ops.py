# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from source.domain_object import Action, Transition

# ---- Q(s,a) from V(s) -------------------------------------------------------
def q_from_v(
    P: Dict[int, Dict[Action, List[Transition]]],
    V: np.ndarray,
    gamma: float,
) -> Dict[int, Dict[Action, float]]:
    """
    Q(s,a) = E[r + γ * V(s')].
    对终止转移，下一步价值记为 0（吸收态）。
    """
    Q: Dict[int, Dict[Action, float]] = {}
    for s, amap in P.items():
        Q[s] = {}
        for a, trans_list in amap.items():
            q = 0.0
            for tr in trans_list:
                q += tr.prob * (tr.reward + gamma * (0.0 if tr.done else V[tr.next_state_id]))
            Q[s][a] = q
    return Q


# ---- 贪心策略 π_greedy(V) 或 π_greedy(Q) -------------------------------------
def greedy_policy_from_q(
    Q: Dict[int, Dict[Action, float]],
    tie_breaker: List[Action],
) -> Dict[int, Dict[Action, float]]:
    """
    返回 one-hot 的确定性贪心策略。
    用 tie_breaker 顺序打破并列。
    """
    pi: Dict[int, Dict[Action, float]] = {}
    for s, amap in Q.items():
        if not amap:
            pi[s] = {Action.STAY: 1.0}
            continue
        max_q = max(amap.values())
        # 收集并列动作
        best_as = [a for a, q in amap.items() if np.isclose(q, max_q)]
        best_a = sorted(best_as, key=lambda x: tie_breaker.index(x))[0]
        pi[s] = {a: (1.0 if a == best_a else 0.0) for a in Action}
    return pi


# ---- E_π[r + γ V(s')] 单步期望备份（供策略评估使用） ---------------------------
def expected_backup_under_pi(
    P: Dict[int, Dict[Action, List[Transition]]],
    pi: Dict[int, Dict[Action, float]],
    V: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    计算 T_π V 的一轮备份： (T_π V)(s) = Σ_a π(a|s) Σ_{s',r} p * (r + γ V(s')).
    """
    nS = len(P)
    V_new = np.zeros(nS, dtype=float)
    for s in range(nS):
        val = 0.0
        amap = P[s]
        dist = pi.get(s, {})
        if not dist:
            V_new[s] = 0.0
            continue
        for a, a_prob in dist.items():
            if a_prob == 0.0:
                continue
            q_sa = 0.0
            for tr in amap[a]:
                q_sa += tr.prob * (tr.reward + gamma * (0.0 if tr.done else V[tr.next_state_id]))
            val += a_prob * q_sa
        V_new[s] = val
    return V_new


# ---- ||T*V - V||_inf --------------------------------------------------------
def bellman_residual_optimality(
    P: Dict[int, Dict[Action, List[Transition]]],
    V: np.ndarray,
    gamma: float,
) -> float:
    """
    计算最优性残差：||T* V - V||_inf = max_s |max_a Q(s,a) - V(s)|.
    """
    nS = len(P)
    res = 0.0
    for s in range(nS):
        amap = P[s]
        if not amap:
            res = max(res, abs(0.0 - V[s]))
            continue
        max_q = -np.inf
        for a, trans_list in amap.items():
            q = 0.0
            for tr in trans_list:
                q += tr.prob * (tr.reward + gamma * (0.0 if tr.done else V[tr.next_state_id]))
            if q > max_q:
                max_q = q
        res = max(res, abs(max_q - V[s]))
    return float(res)


# ---- 策略相等性判断（供 PI 收敛判据） -----------------------------------------
def policy_equal(pi1: Dict[int, Dict[Action, float]], pi2: Dict[int, Dict[Action, float]]) -> bool:
    if pi1.keys() != pi2.keys():
        return False
    for s in pi1:
        d1, d2 = pi1[s], pi2[s]
        if d1.keys() != d2.keys():
            return False
        for a in d1:
            if not np.isclose(d1[a], d2[a]):
                return False
    return True