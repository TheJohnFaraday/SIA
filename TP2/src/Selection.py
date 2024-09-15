from enum import Enum
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from .PlayerAttributes import PlayerAttributes
from .Player import Player
from .PlayerClass import PlayerClass


class SelectionMethods(Enum):
    ELITE = "elite"
    ROULETTE = "roulette"
    UNIVERSAL = "universal"
    BOLTZMANN = "boltzmann"
    DETERMINISTIC_TOURNAMENT = "deterministic_tournament"
    PROBABILISTIC_TOURNAMENT = "probabilistic_tournament"
    RANKING = "ranking"


@dataclass(frozen=True)
class Configuration:
    method: list[SelectionMethods]
    weight: list[Decimal]
    deterministic_tournament_individuals_to_select: int = 0
    probabilistic_tournament_threshold: Decimal = 0
    boltzmann_temperature: Decimal = 0


class Selection:
    def __init__(self, population_sample: int, configuration: Configuration):
        self.__configuration = configuration
        self.__population_sample = population_sample

    def select(self, current_population: list[Player], new_population: list[Player]):
        population = current_population + new_population

        final_population = []

        for index, method in enumerate(self.__configuration.method):
            method_weight = self.__configuration.weight[index]
            selected_population = []
            match method:
                case SelectionMethods.ELITE:
                    selected_population = Selection.elite(
                        population, self.__population_sample
                    )
                case SelectionMethods.ROULETTE:
                    selected_population = Selection.roulette(
                        population, self.__population_sample
                    )
                case SelectionMethods.UNIVERSAL:
                    selected_population = Selection.universal(
                        population, self.__population_sample
                    )
                case SelectionMethods.BOLTZMANN:
                    selected_population = Selection.boltzmann(
                        population,
                        self.__population_sample,
                        self.__configuration.boltzmann_temperature,
                    )
                # case SelectionMethods.DETERMINISTIC_TOURNAMENT:
                #     selected_population = Selection.deterministic_tournament(
                #         population,
                #         self.__population_sample,
                #         self.__configuration.deterministic_tournament_individuals_to_select,
                #     )
                # case SelectionMethods.PROBABILISTIC_TOURNAMENT:
                #     selected_population = Selection.probabilistic_tournament(
                #         population,
                #         self.__population_sample,
                #         self.__configuration.probabilistic_tournament_threshold,
                #     )
                case SelectionMethods.RANKING:
                    selected_population = Selection.ranking(
                        population, self.__population_sample
                    )
                case _:
                    raise RuntimeError("Invalid selection method!")

            if method_weight == Decimal(1):
                return selected_population

            players_to_select = int(round(method_weight * len(selected_population)))
            final_population += selected_population[:players_to_select]

        return final_population

    @staticmethod
    def _calculate_fitness(
        population: list[Player],
    ) -> tuple[list[Decimal], list[Decimal]]:
        """
        Calculates the relative and cumulative fitness for the population.
        Returns a tuple (relative_fitness, cumulative_fitness).
        """
        total_fitness = sum([player.fitness for player in population])
        relative_fitness = [player.fitness / total_fitness for player in population]

        cumulative_fitness = []
        cumulative_sum = Decimal(0.0)
        for fitness in relative_fitness:
            cumulative_sum += fitness
            cumulative_fitness.append(cumulative_sum)

        print(cumulative_fitness)
        return relative_fitness, cumulative_fitness

    @staticmethod
    def _select_by_random_numbers(
        cumulative_fitness: list[Decimal],
        random_numbers: list[Decimal],
        population: list[Player],
    ) -> list[Player]:
        """
        Selects individuals from the population based on a list of random numbers.
        """
        selected_population = []
        for random_number in random_numbers:
            for i in range(1, len(cumulative_fitness) + 1):
                if cumulative_fitness[i - 1] < random_number <= cumulative_fitness[i]:
                    selected_population.append(population[i])
                    break

        return selected_population

    @staticmethod
    def elite(population: list[Player], population_sample_length: int) -> list[Player]:
        """
        Selects the individuals from the population based on their scores
        """
        population_length = len(population)

        population.sort(reverse=True)

        if population_sample_length <= population_length:
            return population[0:population_sample_length]

        population_with_repetition = []
        for i in range(population_sample_length):
            population_with_repetition.append(population[i % population_length])

        return population_with_repetition

    @staticmethod
    def roulette(
        population: list[Player], population_sample_length: int
    ) -> list[Player]:
        """
        Roulette selection: selects individuals from the population based on their scores in a stochastic manner.
        """
        relative_fitness, cumulative_fitness = Selection._calculate_fitness(population)

        random_numbers = list(
            map(Decimal, np.random.uniform(0, 1, population_sample_length))
        )

        selected_population = Selection._select_by_random_numbers(
            cumulative_fitness, random_numbers, population
        )

        return selected_population

    @staticmethod
    def universal(
        population: list[Player], population_sample_length: int
    ) -> list[Player]:
        """
        Universal stochastic selection: selects individuals from the population based on their scores with more uniformity.
        """
        relative_fitness, cumulative_fitness = Selection._calculate_fitness(population)

        r_value = np.random.uniform(0, 1)
        j_values = np.arange(0, population_sample_length)
        random_numbers = list(
            map(Decimal, (r_value + j_values) / population_sample_length)
        )

        selected_population = Selection._select_by_random_numbers(
            cumulative_fitness, random_numbers, population
        )

        return selected_population

    @staticmethod
    def ranking(
        population: list[Player], population_sample_length: int
    ) -> list[Player]:
        """
        Ranking selection: selects individuals from the population based on their rank.
        """
        population_len = len(population)
        sorted_indices = np.argsort([float(player.fitness) for player in population])[
            ::-1
        ].tolist()
        sorted_players = [population[index] for index in sorted_indices]

        # Assign ranks based on sorted fitness
        pseudo_population: list[Player] = []
        for rank, index in enumerate(sorted_indices):
            # ranks[index] = rank + 1
            player = sorted_players[index]
            player.fitness = (population_len - (1 + rank)) / population_len
            pseudo_population.append(player)

        pseudo_relative_fitness, pseudo_cumulative_fitness = (
            Selection._calculate_fitness(pseudo_population)
        )
        random_numbers = list(map(np.random.uniform(0, 1, population_sample_length)))

        selected_population = Selection._select_by_random_numbers(
            pseudo_cumulative_fitness, random_numbers, population
        )

        return selected_population

    @staticmethod
    def boltzmann(
        population: list[Player], population_sample_length: int, temperature: Decimal
    ) -> list[Player]:
        """
        Boltzmann's selection: selects individuals from the population based on a probability
        distribution that depends on their fitness and the current temperature. Higher fitness
        individuals are more likely to be selected as the temperature decreases, allowing for
        an initial phase of exploration followed by exploitation as the algorithm progresses.
        """
        relative_fitness = np.exp(
            [player.fitness / temperature for player in population]
        )

        average_fitness = np.average(relative_fitness)

        pseudo_population = []
        for index, player in enumerate(population):
            pseudo_population.append(
                Player(
                    height=player.height,
                    p_class=player.p_class,
                    p_attr=player.p_attr,
                    fitness=relative_fitness[index] / average_fitness,
                )
            )

        pseudo_relative_fitness, pseudo_cumulative_fitness = (
            Selection._calculate_fitness(pseudo_population)
        )
        random_numbers = list(
            map(Decimal, np.random.uniform(0, 1, population_sample_length))
        )

        selected_population = Selection._select_by_random_numbers(
            pseudo_cumulative_fitness, random_numbers, population
        )

        return selected_population


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
    result = Selection.boltzmann(population3, 3, Decimal(3))

    for player in result:
        print(player.fitness)
