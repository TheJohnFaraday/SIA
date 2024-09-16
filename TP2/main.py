from decimal import Decimal, getcontext, Context
import random

from src.configuration import read_configuration
from src.Cross import Cross
from src.Finish import Finish
from src.Mutation import Mutation
from src.Player import Player, PlayerClass, PlayerAttributes
from src.Selection import Selection
from src.utils import random_numbers_that_sum_n
from src.Replacement import Replacement


context = Context(prec=10)
getcontext().prec = 10


def initial_population(
    player_class: PlayerClass, size: int, max_points: int
) -> list[Player]:
    def player_generator():
        attributes = random_numbers_that_sum_n(
            PlayerAttributes.NUMBER_OF_ATTRIBUTES,
            max_points,
        )
        return Player(
            height=Decimal(
                random.uniform(float(Player.MIN_HEIGHT), float(Player.MAX_HEIGHT))
            ),
            p_class=player_class,
            p_attr=PlayerAttributes(*attributes),
            fitness=Decimal(1),
        )

    return [player_generator() for _ in range(size)]


if __name__ == "__main__":
    configuration = read_configuration()

    random.seed(configuration.random_seed)

    selection = Selection(
        population_sample=configuration.population_sample,
        configuration=configuration.selection,
    )
    crossover = Cross(configuration.genetic.crossover, configuration.points)
    mutation = Mutation(configuration.genetic.mutation, configuration.points)
    replacement = Replacement(configuration.genetic.replacement)
    finish = Finish(configuration.finish)

    history = []
    generation = 0
    population = initial_population(
        configuration.player, configuration.initial_population, configuration.points
    )
    while not finish.done(population):
        new_population = []
        generation += 1

        # Selection
        selected_population = selection.select(population)

        # Crossover
        random.shuffle(selected_population)
        for index, player in enumerate(selected_population):
            crossed = crossover.perform(
                selected_population[index],
                selected_population[(index + 1) % len(selected_population)],
            )
            new_population += crossed

        # Mutation
        mutation.mutate(new_population)

        # Replacement
        population = replacement.replace(selected_population, new_population)

        # TODO: History and metrics
