# -*- coding: utf-8 -*-
# 路径：source/domain_object/sa_expression.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Iterable
import numpy as np

Array = np.ndarray

@dataclass
class RootFindingOracle:
    """
    用于 Robbins–Monro：g(w) 的黑箱 + 有噪声观测  ĝ(w, η) = g(w) + η
    支持标量或向量 w（np.ndarray），噪声可控。
    """
    g: Callable[[Array], Array]                          # 真实 g(w)
    noise_sampler: Optional[Callable[[], Array]] = None  # 采样 η_k；None 表示无噪声
    dim: int = 1                                         # w 的维度（标量用 1）
    projection: Optional[Callable[[Array], Array]] = None  # 可选投影，保证有界（收敛条件常用）

    def observe(self, w: Array) -> Array:
        eta = self.noise_sampler() if self.noise_sampler is not None else 0.0
        return self.g(w) + eta

    def project(self, w: Array) -> Array:
        return self.projection(w) if self.projection is not None else w


@dataclass
class MinimizationOracle:
    """
    用于 SGD：最小化 J(w)=E[f(w, X)]；给出 f 的随机样本以及梯度估计。
    """
    # 采样随机变量 X 的函数
    data_sampler: Callable[[int], Iterable]                      # sampler(batch_size)->batch
    # 单样本（或小批）目标与梯度：fn_grad(w, x_batch)->(loss_scalar, grad_vector)
    fn_grad: Callable[[Array, Iterable], Tuple[float, Array]]
    dim: int = 1
    projection: Optional[Callable[[Array], Array]] = None

    def grad_estimate(self, w: Array, batch) -> Tuple[float, Array]:
        return self.fn_grad(w, batch)

    def project(self, w: Array) -> Array:
        return self.projection(w) if self.projection is not None else w
