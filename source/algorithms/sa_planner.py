# -*- coding: utf-8 -*-
# 路径：source/algorithms/sa_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from source.sa_expression import RootFindingOracle, MinimizationOracle
from source.utils.sa_schedules import rm_step_sequence, gd_fixed, series_diagnosis, moving_average
from source.utils.logger_manager import LoggerManager     # 统一日志/TFBoard
from source.utils.timing import record_time_decorator

Array = np.ndarray

@dataclass
class SAConfig:
    seed: int = 42
    log_dir: str = "logs/ch6"
    use_tensorboard: bool = True
    # 诊断
    ma_window: int = 100

class SAPlanner:
    """
    Chapter 6: Stochastic Approximation & SGD
      - Robbins–Monro root finding
      - GD/BGD/MBGD/SGD for minimization (实质解 ∇J(w)=0 的根)
    """
    def __init__(self, cfg: Optional[SAConfig] = None) -> None:
        self.cfg = cfg or SAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.logger = LoggerManager(self.cfg.log_dir, self.cfg.use_tensorboard)
        self.step_counter = 0

    # ------------------ Robbins–Monro ------------------
    @record_time_decorator("Robbins-Monro")
    def robbins_monro(
        self,
        oracle: RootFindingOracle,
        w0: Array,
        *,
        a0: float = 1.0,
        beta: float = 1.0,
        max_iter: int = 10_000,
        tol: float = 1e-6,
        diag_every: int = 50,
    ) -> Dict[str, Any]:
        """
        w_{k+1} = w_k - a_k * ĝ(w_k),  ĝ(w_k)=g(w_k)+η_k
        收敛充分条件(摘): 0 < c1 <= ||∇g(w)|| <= c2； sum a_k=∞, sum a_k^2<∞；噪声零均值且二阶可积。
        参数 beta∈(0.5,1] 满足步长条件。
        """
        sum_inf, sum_sq = series_diagnosis(a0, beta)
        if not (sum_inf and sum_sq):
            self.logger.log(f"[WARN] a_k=a0/k^{beta} 不满足 RM 的标准充分条件 (sum_inf={sum_inf}, sum_sq_finite={sum_sq}).")

        w = np.array(w0, dtype=float)
        steps = rm_step_sequence(a0=a0, beta=beta)
        residual_hist = []

        for k in range(1, max_iter + 1):
            g_hat = oracle.observe(w)
            w = w - next(steps) * g_hat
            w = oracle.project(w)

            res = float(np.linalg.norm(g_hat))
            residual_hist.append(res)

            if k % diag_every == 0:
                ma = moving_average(residual_hist, self.cfg.ma_window)
                self.logger.add_scalar("RM/residual", res, k)
                if len(ma) > 0:
                    self.logger.add_scalar("RM/residual_ma", ma[-1], k)
                self.logger.log(f"[RM] k={k}, ||g_hat||={res:.3e}")

            if res < tol:
                self.logger.log(f"[RM] Converged at k={k}, ||g_hat||={res:.3e}")
                break

        return {"w": w, "residual_hist": residual_hist}

    # ------------------ Minimization: GD/BGD/MBGD/SGD ------------------
    @record_time_decorator("Minimization")
    def minimize(
        self,
        oracle: MinimizationOracle,
        w0: Array,
        *,
        method: str = "sgd",             # "gd" | "bgd" | "mbgd" | "sgd"
        lr: float = 0.1,                 # GD 固定学习率
        rm_a0: float = 1.0,              # 其余方法用 RM 步长
        rm_beta: float = 0.9,
        batch_size: int = 1,             # MBGD/SGD 用
        max_iter: int = 10_000,
        tol: float = 1e-6,
        diag_every: int = 50,
    ) -> Dict[str, Any]:
        """
        - GD:  使用固定 lr（课件中期望梯度难拿，这里仅作对比）
        - BGD: 使用全数据的梯度（示例中可自定义 data_sampler 返回“全部样本”）
        - MBGD/SGD: 使用 RM 步长（满足 sum a_k=∞ & sum a_k^2<∞），k→∞ 收敛到 ∇J(w)=0 的根（在常见假设下）。
        """
        w = np.array(w0, dtype=float)

        if method.lower() == "gd":
            steps = gd_fixed(lr)
        else:
            # 采用 RM 步长，满足收敛充分条件（0.5<beta<=1）
            sum_inf, sum_sq = series_diagnosis(rm_a0, rm_beta)
            if not (sum_inf and sum_sq):
                self.logger.warning(f"[WARN] RM steps for {method} not ideal: (sum_inf={sum_inf}, sum_sq_finite={sum_sq}).")
            steps = rm_step_sequence(rm_a0, rm_beta)

        loss_hist, grad_norm_hist = [], []

        for k in range(1, max_iter + 1):
            # 采样批次
            if method.lower() == "bgd":
                batch = oracle.data_sampler(-1)  # 约定：返回全量
            elif method.lower() == "mbgd":
                batch = oracle.data_sampler(batch_size)
            elif method.lower() == "sgd":
                batch = oracle.data_sampler(1)
            else:  # gd
                batch = oracle.data_sampler(-1)

            loss, grad = oracle.grad_estimate(w, batch)
            step = next(steps)
            w = oracle.project(w - step * grad)

            gnorm = float(np.linalg.norm(grad))
            loss_hist.append(float(loss))
            grad_norm_hist.append(gnorm)

            if k % diag_every == 0:
                self.logger.add_scalar(f"{method.upper()}/loss", loss_hist[-1], k)
                self.logger.add_scalar(f"{method.upper()}/grad_norm", gnorm, k)
                self.logger.log(f"[{method.upper()}] k={k}, loss={loss_hist[-1]:.6f}, ||grad||={gnorm:.3e}")

            if gnorm < tol:
                self.logger.log(f"[{method.upper()}] Stopping at k={k}, ||grad||={gnorm:.3e}")
                break

        return {
            "w": w,
            "loss_hist": loss_hist,
            "grad_norm_hist": grad_norm_hist,
        }