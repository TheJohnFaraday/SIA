from enum import Enum
from dataclasses import dataclass
from decimal import Decimal

import numpy as np


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
    deterministic_tournament_individuals_to_select: int = 0
    probabilistic_tournament_threshold: Decimal = 0
    boltzmann_temperature: int = 0


class SelectionMethods:
    def __init__(self, configuration: Configuration):
        self.__configuration = configuration

    @staticmethod
    def elite(population: list[float], K: int) -> list[float]:
        """
        Selects the individuals from the population based on their scores
        """
        population_lenght = len(population)

        population.sort(reverse=True)

        if K <= population_lenght:
            return population[0:K]

        population_with_repetition = []
        for i in range(K):
            population_with_repetition.append(population[i % population_lenght])

        return population_with_repetition

    @staticmethod
    def _calculate_fitness(population: list[float]) -> tuple[list[float], list[float]]:
        """
        Calculates the relative and cumulative fitness for the population.
        Returns a tuple (relative_fitness, cumulative_fitness).
        """
        total_fitness = sum(population)
        relative_fitness = [score / total_fitness for score in population]

        cumulative_fitness = []
        cumulative_sum = 0.0
        for fitness in relative_fitness:
            cumulative_sum += fitness
            cumulative_fitness.append(cumulative_sum)

        print(cumulative_fitness)
        return relative_fitness, cumulative_fitness

    @staticmethod
    def _select_by_random_numbers(
        cumulative_fitness: list[float],
        random_numbers: list[float],
        population: list[float],
    ) -> list[float]:
        """
        Selects individuals from the population based on a list of random numbers.
        """
        selected_population = []
        for random_number in random_numbers:
            for i in range(1, len(cumulative_fitness) + 1):
                if cumulative_fitness[i - 1] < random_number <= cumulative_fitness[i]:
                    selected_population.append(population[i])
                    break

        print(random_numbers)
        return selected_population

    @staticmethod
    def roulette(population: list[float], K: int) -> list[float]:
        """
        Roulette selection: selects individuals from the population based on their scores in a stochastic manner.
        """
        relative_fitness, cumulative_fitness = SelectionMethods._calculate_fitness(
            population
        )

        random_numbers = np.random.uniform(0, 1, K)

        selected_population = SelectionMethods._select_by_random_numbers(
            cumulative_fitness, random_numbers, population
        )

        return selected_population

    @staticmethod
    def universal(population: list[float], K: int) -> list[float]:
        """
        Universal stochastic selection: selects individuals from the population based on their scores with more uniformity.
        """
        relative_fitness, cumulative_fitness = SelectionMethods._calculate_fitness(
            population
        )

        r_value = np.random.uniform(0, 1)
        j_values = np.arange(0, K)
        random_numbers = (r_value + j_values) / K

        selected_population = SelectionMethods._select_by_random_numbers(
            cumulative_fitness, random_numbers, population
        )

        return selected_population

    @staticmethod
    def ranking(population: list[float], K: int) -> list[float]:
        """
        Ranking selection: selects individuals from the population based on their rank.
        """
        sorted_indices = np.argsort(population)[::-1]
        ranks = np.zeros_like(sorted_indices, dtype=float)

        # Assign ranks based on sorted fitness
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank + 1

        N = len(population)
        pseudo_population = [(N - rank) / N for rank in ranks]

        pseudo_relative_fitness, pseudo_cumulative_fitness = (
            SelectionMethods._calculate_fitness(pseudo_population)
        )
        random_numbers = np.random.uniform(0, 1, K)

        selected_population = SelectionMethods._select_by_random_numbers(
            pseudo_cumulative_fitness, random_numbers, population
        )

        return selected_population

    @staticmethod
    def boltzmann(population: list[float], K: int, T: float) -> list[float]:
        """
        Boltzmann selection: selects individuals from the population based on a probability
        distribution that depends on their fitness and the current temperature. Higher fitness
        individuals are more likely to be selected as the temperature decreases, allowing for
        an initial phase of exploration followed by exploitation as the algorithm progresses.
        """
        relative_fitness = np.array(np.exp([score / T for score in population]))

        average_fitness = np.average(relative_fitness)

        pseudo_population = np.array(
            [fitness / average_fitness for fitness in relative_fitness]
        )

        pseudo_relative_fitness, pseudo_cumulative_fitness = (
            SelectionMethods._calculate_fitness(pseudo_population)
        )
        random_numbers = np.random.uniform(0, 1, K)

        selected_population = SelectionMethods._select_by_random_numbers(
            pseudo_cumulative_fitness, random_numbers, population
        )

        return selected_population


if __name__ == "__main__":
    population1 = [
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
    population2 = [
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
    population3 = [3, 6, 11, 14, 1]
    # print(SelectionMethods.elite(population, 30))
    print(SelectionMethods.boltzmann(population3, 3, 3))
