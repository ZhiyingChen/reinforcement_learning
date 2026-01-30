# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.mc_planner import MCPlanner, MCConfig
from source.utils.render import render_value_grid, render_policy_grid
import numpy as np

if __name__ == "__main__":
    env = GridWorld(
        height=5,
        width=5,
        target=(3, 2),
        forbidden={(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)},
        start=(0, 0),
        seed=42,
    )

    planner = MCPlanner(env, MCConfig(first_visit=True, seed=2026), log_dir="logs/mc_exploring_starts")
    Q, pi_es = planner.mc_exploring_starts(num_episodes=5000)

    V = np.zeros(len(env.id2s))
    for sid in range(len(env.id2s)):
        q_s = [Q.get(sid, {}).get(a, 0.0) for a in env.allowed_actions(env.id2s[sid])]
        V[sid] = max(q_s) if q_s else 0.0

    render_value_grid(env, V)
    render_policy_grid(env, pi_es)