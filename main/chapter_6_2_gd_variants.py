# -*- coding: utf-8 -*-
# 路径：main/chapter_6_2_gd_variants.py
import numpy as np
from typing import Iterable, Tuple
from source.algorithms.sa_planner import SAPlanner, SAConfig
from source.sa_expression import MinimizationOracle

# 示例任务：均值估计（课件中的等价形式）
# 令 J(w) = E[ 1/2 * || w - X ||^2 ]，则 ∇J(w)= w - E[X]，最优解 w* = E[X]
# 对于批次 x_batch，loss = 1/(2m) Σ ||w - x||^2，grad = (w - mean(x_batch))

def sampler_factory(all_data: np.ndarray):
    rng = np.random.default_rng(2026)
    n = all_data.shape[0]

    def data_sampler(batch_size: int) -> Iterable:
        if batch_size is None or batch_size <= 0:
            return all_data  # BGD: 返回全量
        idx = rng.integers(0, n, size=batch_size)
        return all_data[idx]
    return data_sampler

def fn_grad(w: np.ndarray, x_batch: Iterable) -> Tuple[float, np.ndarray]:
    x = np.asarray(list(x_batch)) if not isinstance(x_batch, np.ndarray) else x_batch
    if x.ndim == 1:  # 标量
        x = x.reshape(-1, 1)
    w = w.reshape(1, -1)
    mean_x = x.mean(axis=0, keepdims=True)
    diff = w - mean_x
    loss = 0.5 * float(((w - x)**2).mean())
    grad = diff.squeeze(0)
    return loss, grad

if __name__ == "__main__":
    # 生成数据：X ~ N(μ, σ^2)
    mu, sigma, n = 2.5, 1.2, 10_000
    data = np.random.default_rng(7).normal(mu, sigma, size=(n, 1))

    oracle = MinimizationOracle(
        data_sampler=sampler_factory(data),
        fn_grad=fn_grad,
        dim=1
    )

    planner = SAPlanner(SAConfig(log_dir="logs/ch6_sgd", use_tensorboard=True))

    w0 = np.array([0.0])

    # 1) BGD：每步用全量数据
    out_bgd = planner.minimize(oracle, w0, method="bgd", rm_a0=1.0, rm_beta=0.9, max_iter=3000, diag_every=100)

    # 2) MBGD：每步用中等批量
    out_mbgd = planner.minimize(oracle, w0, method="mbgd", batch_size=64, rm_a0=1.0, rm_beta=0.9, max_iter=3000, diag_every=100)

    # 3) SGD：每步单样本
    out_sgd = planner.minimize(oracle, w0, method="sgd", rm_a0=1.0, rm_beta=0.9, max_iter=3000, diag_every=100)

    planner.logger.log(f"BGD w_hat: {out_bgd["w"]}")
    planner.logger.log(f"MBGD w_hat: {out_mbgd["w"]}")
    planner.logger.log(f"SGD w_hat: {out_sgd["w"]}")
    planner.logger.log(f"True mean (μ): {mu}")
