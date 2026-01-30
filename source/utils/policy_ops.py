# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from source.domain_object import Action

def normalize_dist(d: Dict[Action, float]) -> Dict[Action, float]:
    s = float(sum(d.values())) or 1.0
    return {a: (p / s) for a, p in d.items()}

def sample_action_from_policy(
    rng: np.random.Generator,
    pi_s: Dict[Action, float],
) -> Action:
    # 以动作枚举的固定顺序保证可复现性
    actions, probs = zip(*sorted(pi_s.items(), key=lambda kv: kv[0].value))
    return rng.choice(list(actions), p=np.array(probs, dtype=float))

def greedy_one_hot_from_q(
    q_s: Dict[Action, float],
    tie_breaker: List[Action],
) -> Dict[Action, float]:
    if not q_s:
        return {Action.STAY: 1.0}
    max_q = max(q_s.values())
    best = [a for a, q in q_s.items() if np.isclose(q, max_q)]
    a_star = sorted(best, key=lambda a: tie_breaker.index(a))[0]
    return {a: (1.0 if a == a_star else 0.0) for a in Action}

def epsilon_greedy_from_q(
    q_s: Dict[Action, float],
    actions: List[Action],
    epsilon: float,
    tie_breaker: List[Action],
) -> Dict[Action, float]:
    """π_ε：最优动作拿到 1-ε + ε/|A|，其他 ε/|A|"""
    if not q_s:
        # 无 Q 时退化为均匀随机
        p = 1.0 / len(actions)
        return {a: p for a in actions}

    max_q = max(q_s.get(a, -np.inf) for a in actions)
    best = [a for a in actions if np.isclose(q_s.get(a, -np.inf), max_q)]
    a_star = sorted(best, key=lambda a: tie_breaker.index(a))[0]

    nA = len(actions)
    base = epsilon / nA
    pi = {a: base for a in actions}
    pi[a_star] = 1.0 - epsilon + base
    return normalize_dist(pi)