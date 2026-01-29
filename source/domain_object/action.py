from enum import Enum
from typing import Dict, Tuple, List, Optional, Iterable



class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

    @staticmethod
    def all() -> List["Action"]:
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY]