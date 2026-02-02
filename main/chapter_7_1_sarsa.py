# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.td_planner import TDPlanner, TDConfig
from source.utils.render import render_policy_grid

if __name__ == "__main__":
    env = GridWorld(
        height=3,
        width=3,
        target=(2, 2),
        forbidden={(1, 1)},
        start=(0, 0),
        seed=42,
    )
    planner = TDPlanner(env, TDConfig(log_dir="logs/ch7_sarsa", use_tensorboard=True))
    Q, pi = planner.sarsa(num_episodes=2000, alpha=0.1, epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.02)
    render_policy_grid(env, pi)