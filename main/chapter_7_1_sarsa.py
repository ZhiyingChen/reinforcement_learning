# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.td_planner import TDPlanner, TDConfig
from source.utils.render import render_policy_grid, render_action_values_grid

if __name__ == "__main__":
    env = GridWorld(
        height=4,
        width=4,
        target=(3, 2),
        forbidden={(1, 1), (1, 2), (2, 2), (3, 1), (3, 3)},
        start=(0, 0),
        seed=7,
    )
    planner = TDPlanner(env, TDConfig(log_dir="logs/ch7_sarsa", use_tensorboard=True))
    Q, pi = planner.sarsa(num_episodes=2000, alpha=0.1, epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.02)
    render_policy_grid(env, pi)
    render_action_values_grid(env, Q, ndigits=2)