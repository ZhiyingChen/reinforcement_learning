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
        start=(0, 0),
        seed=42,
    )

    cfg = PlannerConfig(gamma=0.9, theta=1e-8, eval_theta=1e-10)
    planner = VPPlanner(env, cfg, log_dir="logs/value_iteration")

    # 1) Value Iteration
    V_vi, pi_vi = planner.value_iteration()
    planner.logger.log("\n=== Value Iteration ===")
    render_value_grid(env, V_vi)
    render_policy_grid(env, pi_vi)
    planner.logger.log(f"Residual: {planner.optimality_residual(V_vi):.3e}")

