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

    planner = MCPlanner(env, MCConfig(first_visit=False, seed=8), log_dir="logs/mc_greedy")
    Q, pi_eps = planner.mc_epsilon_greedy(
        num_episodes=8000,
        epsilon=0.2,
        epsilon_decay=0.999,   # 可选：逐步衰减
        min_epsilon=0.02,
        every_visit=True,
    )

    V = np.zeros(len(env.id2s))
    for sid in range(len(env.id2s)):
        q_s = [Q.get(sid, {}).get(a, 0.0) for a in env.allowed_actions(env.id2s[sid])]
        V[sid] = max(q_s) if q_s else 0.0

    render_value_grid(env, V)
    render_policy_grid(env, pi_eps)
