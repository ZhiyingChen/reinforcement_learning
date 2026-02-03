# main/chapter_8_3_dqn_on_policy.py
from source.grid_world import GridWorld
from source.algorithms.tdfa_planner import TDFAPlanner, FAConfig
from source.utils.render import render_policy_grid


if __name__ == "__main__":
    env = GridWorld(height=5, width=5, target=(3,2),
                    forbidden={(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)},
                    start=(0,0), seed=7)
    planner = TDFAPlanner(env, FAConfig(log_dir="logs/ch8_dqn_on"))
    q, tq, pi = planner.dqn_on_policy(num_episodes=1000, epsilon=0.2,
                                      epsilon_decay=0.995, min_epsilon=0.05,
                                      lr=1e-3, hidden=(128,), batch_size=64,
                                      buffer_size=50_000, warmup=1_000,
                                      target_sync_every=500)
    render_policy_grid(env, pi)