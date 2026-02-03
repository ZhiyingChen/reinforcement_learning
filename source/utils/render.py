from ..grid_world import GridWorld
from ..domain_object import Action
import logging
from typing import Dict, List, Tuple, Optional

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
    pad: int = 6,
    h_gap: int = 1,
    v_gap: int = 0,
    missing: Optional[str] = None,
):
    """
    将每个 state 的 5 个 action-value 以“九宫格”形式渲染为带边框的小盒子：
        ┌──────────┐
        │    UP    │
        │ LEFT STAY RIGHT │
        │   DOWN   │
        └──────────┘
    这样每个 state 独立成框，彼此之间用空列/空行分隔，避免混在一起。

    参数
    ----
    env    : GridWorld
    Q      : dict[state_id -> dict[action -> q_value]]
    ndigits: 小数位数
    pad    : 单个数值的定宽（用于对齐）
    h_gap  : 相邻 state（同一行）的水平空列数
    v_gap  : 相邻 state 行块之间的空行数
    missing: 缺失值占位（默认空白；可设为 "--" 或 "NaN"）
    """
    H, W = env.h, env.w
    _get_logger().info("\n[Action Values - Nine Grid (Boxed)]")

    def fmt(v: Optional[float]) -> str:
        if v is None:
            text = "" if missing is None else str(missing)
            return text.rjust(pad)
        return f"{v: .{ndigits}f}".rjust(pad)

    # 计算每个小盒子的宽度（中行包含 LEFT、STAY、RIGHT 三个数值和两个空格）
    mid_width = pad * 3 + 2  # 数字占位+中间两个空格
    box_inner_w = max(mid_width, pad)  # 顶/底/中上/中下均按这个宽度
    horiz = "─" * box_inner_w
    hspace_between = " " * h_gap

    # 逐“网格行”渲染：每个网格行由多个小盒子横向拼接，每个小盒子 5 行（上/中上/中/中下/下）
    for r in range(H):
        # 预先准备该“网格行”的 5 条渲染行（之后拼接每个 state 的盒子）
        row_line_top_border: List[str] = []
        row_line_top:        List[str] = []
        row_line_mid:        List[str] = []
        row_line_bot:        List[str] = []
        row_line_bot_border: List[str] = []

        for c in range(W):
            # 处理特殊格：终点/禁区用大写标识并置于中行居中
            is_target = (env.target is not None and (r, c) == env.target)
            is_forbid = ((r, c) in env.forbidden)

            if is_target or is_forbid:
                label = "T" if is_target else "X"
                top_border = "┌" + horiz + "┐"
                top_line   = "│" + " " * box_inner_w + "│"
                mid_line   = "│" + label.center(box_inner_w) + "│"
                bot_line   = "│" + " " * box_inner_w + "│"
                bot_border = "└" + horiz + "┘"
            else:
                s   = env.s2id[(r, c)]
                q_s = Q.get(s, {})

                up    = q_s.get(Action.UP,    None)
                down  = q_s.get(Action.DOWN,  None)
                left  = q_s.get(Action.LEFT,  None)
                right = q_s.get(Action.RIGHT, None)
                stay  = q_s.get(Action.STAY,  None)

                # 各行内容
                up_str    = fmt(up).center(box_inner_w)
                mid_str   = f"{fmt(left)} {fmt(stay)} {fmt(right)}"
                down_str  = fmt(down).center(box_inner_w)

                top_border = "┌" + horiz + "┐"
                top_line   = "│" + up_str + "│"
                mid_line   = "│" + mid_str + "│"
                bot_line   = "│" + down_str + "│"
                bot_border = "└" + horiz + "┘"

            # 追加到该“网格行”的行缓存
            row_line_top_border.append(top_border)
            row_line_top.append(top_line)
            row_line_mid.append(mid_line)
            row_line_bot.append(bot_line)
            row_line_bot_border.append(bot_border)

        # 把该“网格行”的小盒子横向拼接输出（中间用空格隔开）
        _get_logger().info(hspace_between.join(row_line_top_border))
        _get_logger().info(hspace_between.join(row_line_top))
        _get_logger().info(hspace_between.join(row_line_mid))
        _get_logger().info(hspace_between.join(row_line_bot))
        _get_logger().info(hspace_between.join(row_line_bot_border))

        # 行块之间加空行（视觉分隔）
        for _ in range(v_gap):
            _get_logger().info("")

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
