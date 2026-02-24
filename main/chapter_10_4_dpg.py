from source.grid_world import GridWorld
from source.algorithms.pgac_planner import PGACPlanner, PGACConfig
from source.utils.render import render_policy_grid

if __name__ == "__main__":
    env = GridWorld(
        height=5, width=5, target=(3, 2),
        forbidden={(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)},
        start=(0,0), seed=7, gamma=0.9,
        reward_step=0.0, reward_forbidden=-1.0, reward_target=10.0
    )

    planner = PGACPlanner(env, PGACConfig(log_dir="logs/ch10_dpg"))
    theta, w, pi = planner.dpg(num_episodes=6000, behavior_epsilon=0.4)
    render_policy_grid(env, pi)