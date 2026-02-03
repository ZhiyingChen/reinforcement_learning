# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.td_planner import TDPlanner, TDConfig
from source.utils.render import render_policy_grid, render_action_values_grid, render_value_grid_by_Q

if __name__ == "__main__":
    env = GridWorld(
        height=3,
        width=3,
        target=(2, 2),
        forbidden={(2, 0), (1, 2)},
        start=(0, 0),
        seed=4,
    )

    planner = TDPlanner(env, TDConfig(log_dir="logs/ch7_q_on", use_tensorboard=True))
    Q, pi = planner.q_learning_on_policy(num_episodes=2000, alpha=0.1, epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.02)

    render_policy_grid(env, pi)
    render_value_grid_by_Q(env, Q=Q, Pi=pi)
    render_action_values_grid(env, Q, ndigits=2)