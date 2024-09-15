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

    def attributes_as_list(self):
        """
        Returns a list containing the attributes of this player in the following order:
            [Height, Strength, Dexterity, Intelligence, Endurance, Physique]
        """
        return [
            self.height,
            self.p_attr.strength,
            self.p_attr.dexterity,
            self.p_attr.intelligence,
            self.p_attr.endurance,
            self.p_attr.physique,
        ]
