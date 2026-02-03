# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.mc_planner import MCPlanner, MCConfig
from source.utils.render import render_value_grid_by_Q, render_policy_grid

if __name__ == "__main__":
    env = GridWorld(
        height=3,
        width=3,
        target=(2, 2),
        forbidden={(2, 0), (1, 2)},
        start=(0, 0),
        seed=42,
    )

    planner = MCPlanner(env, MCConfig(first_visit=True, seed=2026), log_dir="logs/mc_exploring_starts")
    Q, pi_es = planner.mc_exploring_starts(num_episodes=5000)

    render_value_grid_by_Q(env, Q=Q, Pi=pi_es)
    render_policy_grid(env, pi_es)