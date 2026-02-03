# main/chapter_8_1_sarsa_linear.py
from source.grid_world import GridWorld
from source.algorithms.tdfa_planner import TDFAPlanner, FAConfig
from source.utils.render import render_policy_grid

if __name__ == "__main__":
    env = GridWorld(height=5, width=5, target=(3,2),
                    forbidden={(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)},
                    start=(0,0), seed=7)
    planner = TDFAPlanner(env, FAConfig(log_dir="logs/ch8_sarsa_linear"))
    w, pi = planner.sarsa_linear(num_episodes=2000, alpha=0.03,
                                 epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.02)
    render_policy_grid(env, pi)