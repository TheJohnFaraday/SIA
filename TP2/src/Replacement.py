from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from random import sample

from .Player import Player
from .PlayerClass import PlayerClass
from .PlayerAttributes import PlayerAttributes


class ReplacementMethod(Enum):
    FILL_ALL = "fill_all"
    FILL_PARENT = "fill_parent"
    GENERATIONAL_GAP = "generational_gap"


@dataclass(frozen=True)
class Configuration:
    method: list[ReplacementMethod]
    weight: list[Decimal]


class Replacement:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

        if len(configuration.method) != len(configuration.weight):
            raise ValueError("'method' and 'weight' length must be the same.")

        weight_sum = sum(configuration.weight)
        if not (weight_sum == Decimal(1)):
            raise ValueError(
                f"the sum of the weights must be 1. Current sum: {weight_sum}"
            )

    @staticmethod
    def fill_all(
        current_population: list[Player], next_population: list[Player]
    ) -> list[Player]:
        # Combine the current and next population
        combined_population = current_population + next_population

        # Select N individuals randomly from the combined population
        new_population = sample(combined_population, len(current_population))

        return new_population


if __name__ == "__main__":
    default_attributes = PlayerAttributes(
        strength=25, dexterity=25, intelligence=20, endurance=15, physique=15
    )

    population1 = [
        Player(
            height=Decimal(1.75),
            p_class=PlayerClass.WARRIOR,
            p_attr=default_attributes,
            fitness=Decimal(fit),
        )
        for fit in [
            0.81,
            0.56,
            0.77,
            0.63,
            0.42,
            0.99,
            0.65,
            0.28,
            0.47,
            0.84,
            0.59,
            0.73,
            0.36,
            0.92,
            0.21,
            0.69,
            0.58,
            0.33,
            0.97,
            0.48,
        ]
    ]
    population2 = [
        Player(
            height=Decimal(1.75),
            p_class=PlayerClass.WARRIOR,
            p_attr=default_attributes,
            fitness=Decimal(fit),
        )
        for fit in [
            81,
            56,
            77,
            63,
            42,
            99,
            65,
            28,
            47,
            84,
            59,
            73,
            36,
            92,
            21,
            69,
            58,
            33,
            97,
            48,
        ]
    ]
    population3 = [
        Player(
            height=Decimal(1.75),
            p_class=PlayerClass.WARRIOR,
            p_attr=default_attributes,
            fitness=Decimal(fit),
        )
        for fit in [3, 6, 11, 14, 1]
    ]
    result = Replacement.fill_all(population1, population2)

    for player in result:
        print(player.fitness)
