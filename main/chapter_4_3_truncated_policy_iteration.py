# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.vp_planner import VPPlanner, PlannerConfig
from source.utils.render import render_value_grid, render_policy_grid


if __name__ == "__main__":
    env = GridWorld(
        height=5,
        width=5,
        target=(3, 2),
        forbidden={(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)},
        start=(4, 4),
        seed=42,
        reward_step=0.0,
        reward_boundary=-1,
        reward_forbidden=-1.0,
        reward_target=10.0,
    )
    cfg = PlannerConfig(gamma=0.9, theta=1e-8, eval_theta=1e-10)
    planner = VPPlanner(env, cfg)

    # 3) Truncated / Modified Policy Iteration（每轮仅评估 3 次）
    V_tpi, pi_tpi = planner.truncated_policy_iteration(eval_sweeps=3)
    planner.logger.log("\n=== Truncated Policy Iteration (k=3) ===")
    render_value_grid(env, V_tpi)
    render_policy_grid(env, pi_tpi)
    planner.logger.log(f"Residual: {planner.optimality_residual(V_tpi):.3e}")