from dataclasses import dataclass
Reward = float

@dataclass(frozen=True)
class Transition:
    prob: float
    next_state_id: int
    reward: Reward
    done: bool
