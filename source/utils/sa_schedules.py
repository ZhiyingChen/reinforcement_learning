# -*- coding: utf-8 -*-
# 路径：source/utils/sa_schedules.py
from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np

def rm_step_sequence(a0: float = 1.0, beta: float = 1.0) -> Iterator[float]:
    """
    Robbins–Monro 步长：a_k = a0 / k^beta，k>=1
    收敛充分条件： sum a_k = ∞ 且 sum a_k^2 < ∞  →  0.5 < beta <= 1
    """
    k = 1
    while True:
        yield a0 / (k ** beta)
        k += 1

def gd_fixed(lr: float) -> Iterator[float]:
    """固定学习率（用于演示/对比）；不满足 RM 充分条件时仅作实践参考。"""
    while True:
        yield lr

def series_diagnosis(a0: float, beta: float) -> Tuple[bool, bool]:
    """
    检查 (sum a_k = ∞, sum a_k^2 < ∞) 的充分条件是否达成（对 a_k=a0/k^beta）。
    返回 (sum_inf, sum_sq_finite)
    """
    sum_inf = (beta <= 1.0)
    sum_sq_finite = (2 * beta > 1.0)
    return sum_inf, sum_sq_finite

def moving_average(x, window=50):
    import numpy as np
    if len(x) == 0:
        return np.array([])
    w = min(window, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid")