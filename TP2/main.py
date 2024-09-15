from decimal import Decimal
from random import shuffle, random, randint

from src.configuration import read_configuration
from src.Cross import Cross
from src.Finish import Finish
from src.Mutation import Mutation
from src.Player import Player, PlayerClass, PlayerAttributes
from src.Selection import Selection
from src.utils import random_numbers_that_sum_n


def initial_population(player_class: PlayerClass, size: int) -> list[Player]:
    return [
        Player(
            height=Decimal(
                min(
                    Player.MAX_HEIGHT,
                    randint(Player.MIN_HEIGHT, Player.MAX_HEIGHT) + random(),
                )
            ),
            p_class=player_class,
            p_attr=PlayerAttributes(
                *random_numbers_that_sum_n(
                    PlayerAttributes.NUMBER_OF_ATTRIBUTES,
                    randint(
                        PlayerAttributes.TOTAL_POINTS_MIN,
                        PlayerAttributes.TOTAL_POINTS_MAX,
                    ),
                )
            ),
            fitness=Decimal(0),
        )
        for _ in range(size)
    ]


if __name__ == "__main__":
    configuration = read_configuration()
    selection = Selection(
        population_sample=configuration.population_sample,
        configuration=configuration.selection,
    )
    crossover = Cross(configuration.genetic.crossover)
    mutation = Mutation(configuration.genetic.mutation)
    finish = Finish(configuration.finish)

    generation = 0
    population = initial_population(
        configuration.player, configuration.initial_population
    )
    while not finish.done():
        new_population = []
        generation += 1

        # Crossover
        shuffle(population)
        for index, player in enumerate(population):
            crossed = crossover.perform(
                population[index], population[(index + 1) % len(population)]
            )
            new_population += crossed

        # Mutation
        mutation.mutate(generation, new_population)

        # Selection
        population = selection.select(population, new_population)

        # TODO: History and metrics
