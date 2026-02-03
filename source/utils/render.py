from ..grid_world import GridWorld
from ..domain_object import Action
import logging
from typing import Dict, List, Tuple

# 统一拿到命名 logger（与 LoggerManager 内一致）
def _get_logger():
    return logging.getLogger("RLLogger")

def render_value_grid(env: GridWorld, V, ndigits=2):
    h, w = env.h, env.w
    _get_logger().info("\n[State Values]")
    for r in range(h):
        row = []
        for c in range(w):
            s = env.s2id[(r, c)]
            if env.target is not None and (r, c) == env.target:
                row.append(" T ")
            elif (r, c) in env.forbidden:
                row.append(" XX")
            else:
                row.append(f"{V[s]: .{ndigits}f}")
        _get_logger().info(" | ".join(row))

def render_action_values_grid(
    env: GridWorld,
    Q: Dict[int, Dict[Action, float]],
    ndigits: int = 2,
    order: Tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY),
    cell_sep: str = " | ",
    pad: int = 5
):
    """
    将每个 state 的 5 个 action value 可视化到网格中。
    - 显示顺序默认为：上、下、左、右、停留（可用 order 自定义）
    - 每个 cell 输出为定宽数字，便于对齐
    - 终点(Target)显示 'T'，禁区显示 'X'

    参数
    ----
    env : GridWorld
    Q   : dict[state_id -> dict[action -> q_value]]
    ndigits : 小数位数
    order   : 展示的动作顺序（默认：上、下、左、右、停留）
    cell_sep: 各 cell 之间的分隔符
    pad     : 单个数值的宽度（用于对齐）
    """
    h, w = env.h, env.w
    _get_logger().info("\n[Action Values]")
    # 准备每个 state 的一行字符串（按顺序拼好）
    for r in range(h):
        row_cells: List[str] = []
        for c in range(w):
            # 终点/禁区直接标记
            if env.target is not None and (r, c) == env.target:
                row_cells.append("  T  ")
                continue
            if (r, c) in env.forbidden:
                row_cells.append("  X  ")
                continue

            s = env.s2id[(r, c)]
            q_s = Q.get(s, {})  # 可能还没有学习到该状态

            # 将指定顺序的 5 个动作的值取出，若不存在则用 0.0 或 None 兜底
            values = []
            for a in order:
                v = q_s.get(a, None)
                if v is None:
                    # 你也可以换成 0.0；这里用 None 更直观显示未定义
                    values.append(" " * pad)
                else:
                    values.append(f"{v: .{ndigits}f}".rjust(pad))

            # 组合为一个 cell：例如 " 0.15  0.02  0.00 -0.03  0.10"
            cell_str = " ".join(values)
            row_cells.append(cell_str)

        # 行与行之间用 cell_sep 分隔
        _get_logger().info(cell_sep.join(row_cells))

def render_policy_grid(env: GridWorld, pi):
    arrow = {Action.UP:"↑", Action.RIGHT:"→", Action.DOWN:"↓", Action.LEFT:"←", Action.STAY:"·"}
    _get_logger().info("\n[Policy]")
    for r in range(env.h):
        row = []
        for c in range(env.w):
            if env.target is not None and (r, c) == env.target:
                row.append("T"); continue
            if (r, c) in env.forbidden:
                row.append("X"); continue
            s = env.s2id[(r, c)]
            a_star = max(pi[s].items(), key=lambda kv: kv[1])[0] if pi.get(s) else Action.STAY
            row.append(arrow[a_star])
        _get_logger().info(" ".join(row))
