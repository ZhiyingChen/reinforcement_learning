from ..grid_world import GridWorld
from ..domain_object import Action


def render_value_grid(env: GridWorld, V, ndigits=2):
    h, w = env.h, env.w
    print("\n[State Values]")
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
        print(" | ".join(row))

def render_policy_grid(env: GridWorld, pi):
    arrow = {Action.UP:"↑", Action.RIGHT:"→", Action.DOWN:"↓", Action.LEFT:"←", Action.STAY:"·"}
    print("\n[Policy]")
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
        print(" ".join(row))