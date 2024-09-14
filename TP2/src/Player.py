from dataclasses import dataclass
from typing import Decimal
from .PlayerClass import PlayerClass
from .PlayerAttributes import PlayerAttributes


@dataclass
class Player:
    height: Decimal
    p_class: PlayerClass
    p_attr: PlayerAttributes
    fitness: Decimal
