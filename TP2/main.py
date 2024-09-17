from decimal import Decimal, getcontext, Context
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.configuration import read_configuration
from src.Cross import Cross
from src.EVE import EVE
from src.Finish import Finish
from src.Mutation import Mutation
from src.Player import Player, PlayerClass, PlayerAttributes
from src.Selection import Selection
from src.utils import random_numbers_that_sum_n
from src.Replacement import Replacement


context = Context(prec=10)
getcontext().prec = 10

CUSTOM_PALETTE = [
    "#508fbe",
    "#f37120",
    "#4baf4e",
    "#f2cb31",
    "#c178ce",
    "#cd4745",
]
GREY = "#6f6f6f"
LIGHT_GREY = "#bfbfbf"

plt.style.use(
    {
        "axes.prop_cycle": plt.cycler(color=CUSTOM_PALETTE),  # Set palette
        "axes.spines.top": False,  # Remove spine (frame)
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.edgecolor": LIGHT_GREY,
        "axes.titleweight": "normal",  # Optional: ensure title weight is normal (not bold)
        "axes.titlelocation": "center",  # Center the title by default
        "axes.titlecolor": GREY,  # Set title color
        "axes.labelcolor": GREY,  # Set labels color
        "axes.labelpad": 10,
        "xtick.bottom": False,  # Remove ticks on the X axis
        "ytick.labelcolor": GREY,  # Set Y ticks color
        "ytick.color": GREY,  # Set Y label color
        "savefig.dpi": 128,
        "legend.frameon": False,
        "legend.labelcolor": GREY,
    }
)


def initial_population(
    player_class: PlayerClass, size: int, max_points: int
) -> list[Player]:
    def player_generator():
        height = Decimal(
            random.uniform(float(Player.MIN_HEIGHT), float(Player.MAX_HEIGHT))
        )
        attributes = random_numbers_that_sum_n(
            PlayerAttributes.NUMBER_OF_ATTRIBUTES,
            max_points,
        )
        player_attributes = PlayerAttributes(*attributes)
        fitness = EVE(
            height=height, p_class=player_class, attributes=player_attributes
        ).performance
        return Player(
            height=height,
            p_class=player_class,
            p_attr=player_attributes,
            fitness=Decimal(fitness),
        )

    return [player_generator() for _ in range(size)]


def calculate_population_variance(population: list[Player]):
    df = pd.DataFrame(
        {
            "height": [float(individual.height) for individual in population],
            "strength": [
                float(individual.p_attr.strength) for individual in population
            ],
            "dexterity": [
                float(individual.p_attr.dexterity) for individual in population
            ],
            "intelligence": [
                float(individual.p_attr.intelligence) for individual in population
            ],
            "endurance": [
                float(individual.p_attr.endurance) for individual in population
            ],
            "physique": [
                float(individual.p_attr.physique) for individual in population
            ],
        }
    )

    variances = df.var()
    mean_variance = variances.mean()

    return mean_variance


def calculate_historic_variance(population_history: list[list[Player]]):
    generational_variance = []
    for population in population_history:
        generational_variance.append(calculate_population_variance(population))

    variance_df = pd.DataFrame(generational_variance, columns=["Variance"])
    return variance_df


def get_historic_fittest_df(population_history: list[list[Player]]):
    fittest_by_generation = []

    for population in population_history:
        population_fitness = [float(individual.fitness) for individual in population]
        fittest_player = max(population, key=lambda p: p.fitness)

        player_max_fitness = max(population_fitness)
        less_fit_player = min(population_fitness)
        mean_fitness = np.mean(population_fitness)
        fittest_by_generation.append(
            (
                player_max_fitness,
                less_fit_player,
                mean_fitness,
                fittest_player.p_class.name,
                *fittest_player.attributes_as_list(),
            )
        )

    fittest_df = pd.DataFrame(
        fittest_by_generation,
        columns=[
            "Maximum fit",
            "Minimum fit",
            "Average fit",
            "Class",
            "Height",
            "Strength",
            "Dexterity",
            "Intelligence",
            "Endurance",
            "Physique",
        ],
    )
    return fittest_df


def standard_deviation_df(population_history: list[list[Player]]):
    std_by_generation = [
        np.std([float(individual.fitness) for individual in population])
        for population in population_history
    ]
    return pd.DataFrame(std_by_generation, columns=["std"])


def plot_results(df: pd.DataFrame, last_gen: [Player]):
    def population_diversity():
        fig, ax = plt.subplots()

        ax.plot(df.index.values, df["Variance"], color=CUSTOM_PALETTE[0])

        ax.set_ylabel("Diversidad")
        ax.set_xlabel("Generación")
        ax.set_title("Diversidad poblacional")
        plt.savefig("plots/population_diversity.png")

    def population_fitness():
        fig, ax = plt.subplots()

        # ax.plot(df.index.values, df["Variance"], color=CUSTOM_PALETTE[0])
        ax.plot(
            df.index.values,
            df["Average fit"],
            label="Average Fit",
            color=CUSTOM_PALETTE[0],
        )
        ax.fill_between(
            df.index.values,
            df["Average fit"] - df["std"],
            df["Average fit"] + df["std"],
            color=CUSTOM_PALETTE[0],
            alpha=0.2,
        )

        # Plot "Maximum fit"
        ax.plot(
            df.index.values,
            df["Maximum fit"],
            label="Maximum Fit",
            color=CUSTOM_PALETTE[2],
        )
        ax.fill_between(
            df.index.values,
            df["Maximum fit"] - df["std"],
            df["Maximum fit"] + df["std"],
            color=CUSTOM_PALETTE[2],
            alpha=0.2,
        )

        # Plot "Minimum fit"
        ax.plot(
            df.index.values,
            df["Minimum fit"],
            label="Minimum Fit",
            color=CUSTOM_PALETTE[1],
        )
        ax.fill_between(
            df.index.values,
            df["Minimum fit"] - df["std"],
            df["Minimum fit"] + df["std"],
            color=CUSTOM_PALETTE[1],
            alpha=0.2,
        )

        # Labeling the axes and title
        ax.set_xlabel("Generación")
        ax.set_ylabel("Fitness")
        ax.set_title("Evolución del Fitness")
        ax.legend()

        plt.savefig("plots/population_fitness.png")

    def fittest_attributes():
        latest_row = df.iloc[-1]  # Fittest player
        attributes = [
            "Height",
            "Strength",
            "Dexterity",
            "Intelligence",
            "Endurance",
            "Physique",
        ]
        player_attributes = latest_row[attributes]

        fig, ax = plt.subplots()

        bars = ax.bar(
            attributes, player_attributes, color=CUSTOM_PALETTE[: len(CUSTOM_PALETTE)]
        )
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}" if int(height) != height else f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=GREY,
                fontweight="bold",
            )

        ax.set_ylabel("")

        ax.set_xlabel("Atributos")
        plt.xticks(rotation=25)

        ax.set_title("Atributos maximales")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.savefig("plots/player_attributes.png")

    def best_attributes():
        df2 = pd.DataFrame(
            {
                "height": [float(individual.height) for individual in last_gen],
                "strength": [
                    float(individual.p_attr.strength) for individual in last_gen
                ],
                "dexterity": [
                    float(individual.p_attr.dexterity) for individual in last_gen
                ],
                "intelligence": [
                    float(individual.p_attr.intelligence) for individual in last_gen
                ],
                "endurance": [
                    float(individual.p_attr.endurance) for individual in last_gen
                ],
                "physique": [
                    float(individual.p_attr.physique) for individual in last_gen
                ],
            }
        )
        means = df2.mean()
        stds = df2.std()
        attributes = [
            "Height",
            "Strength",
            "Dexterity",
            "Intelligence",
            "Endurance",
            "Physique",
        ]

        fig, ax = plt.subplots()

        bars = ax.bar(
            attributes, means, yerr=stds, color=CUSTOM_PALETTE[: len(CUSTOM_PALETTE)]
        )
        height = bars[0].get_height()
        plt.text(
            bars[0].get_x() + bars[0].get_width() / 2,
            height,
            f"{height:.2f}" if int(height) != height else f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=GREY,
            fontweight="bold",
        )
        ax.set_ylabel("Atributos")
        plt.xticks(rotation=25)

        ax.set_title("Atributos de la última generación")
        plt.savefig("plots/last_gen_attributes.png")

    population_diversity()
    population_fitness()
    fittest_attributes()
    best_attributes()


if __name__ == "__main__":
    configuration = read_configuration()

    random.seed(configuration.random_seed)

    selection = Selection(
        population_sample=configuration.population_sample,
        configuration=configuration.selection,
    )
    crossover = Cross(configuration.genetic.crossover, configuration.points)
    mutation = Mutation(configuration.genetic.mutation, configuration.points)
    replacement = Replacement(configuration.replacement)
    finish = Finish(configuration.finish)

    generation = 0
    population = initial_population(
        configuration.player, configuration.initial_population, configuration.points
    )
    history = [population]
    while not finish.done(population, generation):
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

        population.sort(key=lambda player: player.fitness)
        history.append(population)

    print(f"DONE!!! Reason: {finish.reason}")
    print(f"Fittest individual: {population[0]}")
    print(f"Generations: {generation}")

    variance_df = calculate_historic_variance(history)
    fittest_df = get_historic_fittest_df(history)
    std_df = standard_deviation_df(history)
    final_df = pd.concat([variance_df, fittest_df, std_df], axis=1)

    print(final_df)

    if configuration.plot:
        plot_results(final_df, history[len(history) - 1])
