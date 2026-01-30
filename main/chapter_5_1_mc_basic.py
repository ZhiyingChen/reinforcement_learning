# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.mc_planner import MCPlanner, MCConfig
from source.utils.render import render_value_grid, render_policy_grid

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

    # 从 Q 提取 V(s) = max_a Q(s,a)（用于渲染）
    import numpy as np
    V = np.zeros(len(env.id2s))
    for sid in range(len(env.id2s)):
        q_s = [Q.get(sid, {}).get(a, 0.0) for a in env.allowed_actions(env.id2s[sid])]
        V[sid] = max(q_s) if q_s else 0.0

    render_value_grid(env, V)
    render_policy_grid(env, pi_star)
