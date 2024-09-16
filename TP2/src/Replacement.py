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

    @staticmethod
    def fill_parent(
            current_population: list[Player], next_population: list[Player]
    ) -> list[Player]:
        K = len(next_population)
        N = len(current_population)

        if K > N:
            # If there are more children than current population, select N out of K
            new_population = sample(next_population, N)
        else:
            # If there are fewer or equal children than current population,
            # combine K children with (N - K) individuals from the current population
            new_population = next_population.copy()
            if N > K:
                remaining_spots = N - K
                additional_individuals = sample(current_population, remaining_spots)
                new_population.extend(additional_individuals)
            # Ensure we return only N individuals
            new_population = sample(new_population, N)

        return new_population

    @staticmethod
    def generational_gap(
            current_population: list[Player], next_population: list[Player], gap: Decimal
    ) -> list[Player]:
        N = len(current_population)
        num_to_retain = int((1 - gap) * N)
        num_to_replace = N - num_to_retain

        # Ensure num_to_replace does not exceed the size of next_population
        num_to_replace = min(num_to_replace, len(next_population))

        # Select individuals to retain from the current population
        retained_individuals = sample(current_population, num_to_retain)
        # Select individuals to replace from the new population
        new_individuals = sample(next_population, num_to_replace)

        # Combine both to form the new population
        new_population = retained_individuals + new_individuals

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
    result = Replacement.generational_gap(population2, population3, Decimal(0.3))

    for player in result:
        print(player.fitness)
