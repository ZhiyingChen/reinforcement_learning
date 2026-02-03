# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.mc_planner import MCPlanner, MCConfig
from source.utils.render import render_value_grid_by_Q, render_policy_grid

if __name__ == "__main__":
    env = GridWorld(
        height=3,
        width=3,
        target=(2, 2),
        forbidden={(1, 1)},
        start=(0, 0),
        seed=42,
    )

    planner = MCPlanner(env, MCConfig(first_visit=True, seed=123), log_dir="logs/mc_basic")
    Q, pi_star = planner.mc_basic(episodes_per_pair=4, outer_iters=6)

    render_value_grid_by_Q(env, Q=Q, Pi=pi_star)
    render_policy_grid(env, pi_star)
