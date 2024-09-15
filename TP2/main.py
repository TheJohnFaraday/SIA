from random import shuffle

from src.configuration import read_configuration
from src.Cross import Cross
from src.Finish import Finish
from src.Mutation import Mutation
from src.Player import Player
from src.Selection import Selection


def initial_population() -> list[Player]:
    pass


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
    population = initial_population()
    while not finish.done():
        new_population = []
        generation += 1

        # Crossover
        shuffle(population)
        for index, player in enumerate(population):
            crossed = crossover.perform(population[index], population[(index+1) % len(population)])
            new_population += crossed

        # Mutation
        mutation.mutate(generation, new_population)

        # Selection
        population = selection.select(population, new_population)

        # TODO: History and metrics
