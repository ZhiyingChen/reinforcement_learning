# -*- coding: utf-8 -*-
# 路径：main/chapter_6_1_robbins_monro.py
import numpy as np
from source.algorithms.sa_planner import SAPlanner, SAConfig
from source.sa_expression import RootFindingOracle

if __name__ == "__main__":
    # 课件示例：g(w) = tanh(w - 1)，真根 w* = 1
    def g(w: np.ndarray) -> np.ndarray:
        return np.tanh(w - 1.0)

    # 可选噪声：η ~ N(0, σ^2)
    sigma = 0.0
    noise = (lambda: np.random.default_rng(123).normal(0.0, sigma)) if sigma > 0 else None

    oracle = RootFindingOracle(g=g, noise_sampler=noise, dim=1)
    planner = SAPlanner(SAConfig(log_dir="logs/ch6_rm", use_tensorboard=True))

    out = planner.robbins_monro(
        oracle, w0=np.array([3.0]),
        a0=1.0, beta=1.0, max_iter=5000, tol=1e-6, diag_every=100
    )

    planner.logger.log(f"Estimated root: {out['w']}")