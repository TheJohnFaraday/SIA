from dataclasses import dataclass
from decimal import Decimal

from .PlayerClass import PlayerClass
from .PlayerAttributes import PlayerAttributes


@dataclass
class Player:
    height: Decimal
    p_class: PlayerClass
    p_attr: PlayerAttributes
    fitness: Decimal

    def __lt__(self, other: "Player"):
        return self.fitness < other.fitness
