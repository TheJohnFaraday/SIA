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

        if len(configuration.method) != len(configuration.weight):
            raise ValueError("'method' and 'weight' length must be the same.")

        weight_sum = sum(configuration.weight)
        if not (weight_sum == Decimal(1)):
            raise ValueError(f"the sum of the weights must be 1. Current sum: {weight_sum}")

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
                case SelectionMethods.DETERMINISTIC_TOURNAMENT:
                    selected_population = Selection.deterministic_tournament(
                        population,
                        self.__population_sample,
                        self.__configuration.deterministic_tournament_individuals_to_select,
                    )
                case SelectionMethods.PROBABILISTIC_TOURNAMENT:
                    selected_population = Selection.probabilistic_tournament(
                        population,
                        self.__population_sample,
                        self.__configuration.probabilistic_tournament_threshold,
                    )
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
        for idx, random_number in enumerate(random_numbers):
            print(f"Random number {idx + 1}: {random_number}\n")

            # Handle case where random_number is between 0 and the first cumulative_fitness value
            if random_number <= cumulative_fitness[0]:
                selected_population.append(population[0])
                print(
                    f"The random number {random_number} falls into the interval [0, {cumulative_fitness[0]}], "
                    f"so the selected individual is {population[0]}\n"
                )
                continue

            # Iterate over the rest of the intervals
            for i in range(1, len(cumulative_fitness)):
                if cumulative_fitness[i - 1] < random_number <= cumulative_fitness[i]:
                    print(
                        f"The random number {random_number} falls into the interval [{cumulative_fitness[i - 1]}, "
                        f"{cumulative_fitness[i]}], so the selected individual is {population[i]}\n"
                    )
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
        # Create a copy of the population and sort it by fitness in descending order
        sorted_population = sorted(
            population, key=lambda player: player.fitness, reverse=True
        )

        # Calculate pseudo fitness based on ranks
        pseudo_population = [
            Player(
                height=player.height,
                p_class=player.p_class,
                p_attr=player.p_attr,
                fitness=(Decimal(population_len) - i)
                / Decimal(population_len),  # Normalized rank-based fitness
            )
            for i, player in enumerate(sorted_population)
        ]

        pseudo_relative_fitness, pseudo_cumulative_fitness = (
            Selection._calculate_fitness(pseudo_population)
        )

        random_numbers = list(
            map(Decimal, np.random.uniform(0, 1, population_sample_length))
        )

        selected_population = Selection._select_by_random_numbers(
            pseudo_cumulative_fitness, random_numbers, sorted_population
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

    @staticmethod
    def deterministic_tournament(
        population: list[Player],
        population_sample_length: int,
        tournament_participants: int,
    ) -> list[Player]:
        """
        Deterministic tournament selection: randomly selects a group of individuals from the population,
        then selects the best individual based on fitness from that group. Repeats until the desired
        population size is achieved.
        """
        selected_population = []
        for i in range(population_sample_length):
            # Randomly select participants for the tournament
            tournament = np.random.choice(
                population, tournament_participants, replace=False
            )
            print(
                f"Tournament {i + 1}: Participants: {[player.fitness for player in tournament]}"
            )

            # Find the winner (individual with the highest fitness)
            winner = max(tournament, key=lambda player: player.fitness)
            print(f"Winner of tournament {i + 1}: {winner.fitness}\n")

            selected_population.append(winner)

        return selected_population

    @staticmethod
    def probabilistic_tournament(
        population: list[Player],
        population_sample_length: int,
        threshold: Decimal,
    ) -> list[Player]:
        """
        Probabilistic tournament selection: randomly selects two individuals from the population
        and chooses the more fit individual with a probability determined by the threshold value.
        If the random value exceeds the threshold, the less fit individual is chosen. This process
        is repeated until the desired number of individuals is selected.
        """
        selected_population = []

        # Repeat the selection process until we have selected the desired number of individuals
        for i in range(population_sample_length):
            participants = np.random.choice(population, 2, replace=False)

            print(
                f"Tournament {i + 1}: Participants: {[player.fitness for player in participants]}"
            )

            r = np.random.uniform(0, 1)

            # Compare the random value to the threshold to determine the winner
            if r < threshold:
                # Select the more fit individual
                winner = max(participants, key=lambda player: player.fitness)
            else:
                # Select the less fit individual
                winner = min(participants, key=lambda player: player.fitness)

            print(f"Winner of tournament {i + 1}: {winner.fitness}\n")

            selected_population.append(winner)

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
    result = Selection.probabilistic_tournament(population2, 5, Decimal(0.5))

    for player in result:
        print(player.fitness)
