from dataclasses import dataclass
from decimal import Decimal

from .PlayerClass import PlayerClass
from .PlayerAttributes import PlayerAttributes


@dataclass
class Player:
    MIN_HEIGHT = Decimal("1.3")
    MAX_HEIGHT = Decimal("2.0")

    height: Decimal
    p_class: PlayerClass
    p_attr: PlayerAttributes
    fitness: Decimal

    def __lt__(self, other: "Player"):
        return self.fitness < other.fitness
