from source.grid_world import GridWorld

# -----------------------------
# 用法示例
# -----------------------------
if __name__ == "__main__":
    # 4x4 网格；target 在 (0,3)，forbidden 在 {(1,1),(2,3)}
    env = GridWorld(
        height=4,
        width=4,
        target=(0, 3),
        forbidden={(1, 1), (2, 3)},
        start=(3, 0),
        seed=42,
    )

    # 采样运行
    s = env.reset()
    done = False
    total_r = 0.0
    while not done:

        ns, r, done, info = env.step()  # 不传 action -> 按 π(a|s)
        total_r += r
        s = ns
    print("Episode return:", total_r)

    # 拿到 Gym 风格 P，便于策略/价值迭代
    P = env.get_P()
    # P[state_id][Action.UP] -> List[Transition(prob, next_state_id, reward, done)]