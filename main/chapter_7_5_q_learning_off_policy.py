# -*- coding: utf-8 -*-
from source.grid_world import GridWorld
from source.algorithms.td_planner import TDPlanner, TDConfig
from source.utils.render import render_policy_grid, render_action_values_grid, render_value_grid_by_Q

if __name__ == "__main__":
    env = GridWorld(
        height=5,
        width=5,
        target=(3, 2),
        forbidden={(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)},
        start=(0, 0),
        seed=42,
        reward_step=0.0,
        reward_boundary=-1,
        reward_forbidden=-1.0,
        reward_target=10.0,
    )

    planner = TDPlanner(env, TDConfig(log_dir="logs/ch7_q_off", use_tensorboard=True))
    Q, pi_target = planner.q_learning_off_policy(num_episodes=2000, alpha=0.1, behavior_epsilon=0.3, target_epsilon=0.05)

    render_policy_grid(env, pi_target)
    render_value_grid_by_Q(env, Q=Q, Pi=pi_target)
    render_action_values_grid(env, Q, ndigits=2)