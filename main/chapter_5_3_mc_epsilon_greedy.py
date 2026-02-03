# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.mc_planner import MCPlanner, MCConfig
from source.utils.render import render_value_grid_by_Q, render_policy_grid
import numpy as np

if __name__ == "__main__":
    env = GridWorld(
        height=3,
        width=3,
        target=(2, 2),
        forbidden={(2, 0), (1, 2)},
        start=(0, 0),
        seed=42,
    )

    planner = MCPlanner(env, MCConfig(first_visit=False, seed=8), log_dir="logs/mc_greedy")
    Q, pi_eps = planner.mc_epsilon_greedy(
        num_episodes=8000,
        epsilon=0.2,
        epsilon_decay=0.999,   # 可选：逐步衰减
        min_epsilon=0.02,
        every_visit=True,
    )

    render_value_grid_by_Q(env, Q=Q, Pi=pi_eps)
    render_policy_grid(env, pi_eps)
