from ..grid_world import GridWorld
from ..domain_object import Action
import logging

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