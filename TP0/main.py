import json
import sys

import numpy as np
import pandas as pd

from enum import Enum
from matplotlib import pyplot as plt
from dataclasses import dataclass

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon

POKEMONS_CONFIG = "pokemon.json"


class Pokeballs(Enum):
    POKEBALL = "pokeball"
    ULTRA_BALL = "ultraball"
    HEAVY_BALL = "heavyball"
    FAST_BALL = "fastball"


class HP_LEVELS(Enum):
    HP_100 = 1
    HP_80 = 0.8
    HP_60 = 0.6
    HP_40 = 0.4
    HP_20 = 0.2


@dataclass
class CatchesByPokeball:
    pokemon: Pokemon
    ball: Pokeballs
    catches: list[int]


@dataclass
class CatchesByPokeballWithHP(CatchesByPokeball):
    hp: float


@dataclass(frozen=True, eq=True)
class PokeballMean:
    ball: Pokeballs
    mean: float


def create_ideal_pokemon(factory: PokemonFactory, pokemon: str) -> Pokemon:
    return factory.create(pokemon, 100, StatusEffect.NONE, 1)


def catch_with_pokeball(pokemon: Pokemon, ball: Pokeballs, times: int) -> CatchesByPokeball:
    catches_with_ball: list[int] = []
    for _ in range(times):
        catches_with_ball.append(1 if attempt_catch(pokemon, ball.value)[0] else 0)

    return CatchesByPokeball(pokemon, ball, catches_with_ball)


def catch_with_all_pokeballs(pokemon: Pokemon, times: int) -> list[CatchesByPokeball]:
    catches: list[CatchesByPokeball] = []
    for ball in Pokeballs:
        catches.append(catch_with_pokeball(pokemon, ball, times))

    return catches


def catch_with_pokeball_with_hp(pokemon: Pokemon, times: int, hp: float) -> list[CatchesByPokeballWithHP]:
    catch = catch_with_pokeball(pokemon, Pokeballs.POKEBALL, times)
    extended_catch = CatchesByPokeballWithHP(catch.pokemon, catch.ball, catch.catches, hp)

    return extended_catch


def get_pokemons():
    pokemons: set[str] = set()
    with open(POKEMONS_CONFIG) as f:
        pokemon_json = json.load(f)
        for name, _ in pokemon_json.items():
            pokemons.add(name)

    return list(pokemons)


def pandas_aggregate_1a(catches: list[CatchesByPokeball]):
    data = [
        {
            "pokemon": catch.pokemon.name,
            "ball": catch.ball.value,
            "catches": np.sum(catch.catches),
            "throws": len(catch.catches),
        }
        for catch in catches
    ]
    df = pd.DataFrame(data).sort_values(by=["ball", "pokemon"]).reset_index(drop=True)

    grouped = df.groupby("ball").agg(
        catches=("catches", "sum"), throws=("throws", "sum")
    )
    grouped["mean"] = grouped["catches"] / grouped["throws"]
    grouped = grouped.sort_values(by=["ball"])

    return df, grouped


def pandas_aggregate_1b(catches: list[CatchesByPokeball]):
    data = [
        {
            "pokemon": catch.pokemon.name,
            "ball": catch.ball.value,
            "catches": np.sum(catch.catches),
            "throws": len(catch.catches),
        }
        for catch in catches
    ]
    df = pd.DataFrame(data).sort_values(by=["pokemon", "ball"]).reset_index(drop=True)
    df["mean"] = np.divide(df["catches"], df["throws"])

    pokeball_by_pokemon = df[df["ball"] == "pokeball"].set_index("pokemon")
    df["relative_to_pokeball"] = df.apply(
        lambda row: (
            row["mean"] / pokeball_by_pokemon.loc[row["pokemon"]]["mean"]
            if row["ball"] != "pokeball"
            else 1.0
        ),
        axis="columns",
    )

    return df


def plot_1a(df: pd.DataFrame):
    fig, ax = plt.subplots()

    bar_colors = ["tab:red", "#0075BE", "#FFCC00", "tab:orange"]

    ax.bar(df.index.values, df["mean"], color=bar_colors)

    ax.set_ylabel("Probabilidad de captura promedio")
    ax.set_xlabel("Pokeball")
    ax.set_title("Probabilidad de captura promedio por Pokeball")

    plt.show()


def plot_1b(df: pd.DataFrame):
    fig, ax = plt.subplots(layout="constrained")

    bar_colors = ["tab:red", "#0075BE", "#FFCC00", "tab:orange"]

    pokemons = df["pokemon"].unique().tolist()
    balls = df["ball"].unique().tolist()

    x = np.arange(len(pokemons))
    bar_width = 0.20
    multiplier = 0

    color_map = {ball: bar_colors[i % len(bar_colors)] for i, ball in enumerate(balls)}

    values = {}
    for ball in df["ball"].unique():
        values[ball] = tuple(df[df["ball"] == ball]["relative_to_pokeball"])

    for ball, means in values.items():
        offset = bar_width * multiplier
        ax.bar(x + offset, means, bar_width, label=ball, color=color_map[ball])
        multiplier += 1

    ax.set_ylabel("Efectividad de captura relativa")
    ax.set_xlabel("Pokeballs agrupadas por Pokemon")
    ax.set_title("Efectividad de captura de cada Pokemon relativa a Pokeball")

    ax.set_xticks(x + bar_width, pokemons)
    ax.legend(loc="upper right", ncols=2)

    plt.show()


def plot_2b(df: pd.DataFrame):
    fig, ax = plt.subplots(layout="constrained")

    bar_colors = ["tab:red", "#0075BE", "#FFCC00", "tab:orange"]

    pokemons = df["pokemon"]


def ej2b():
    catches: list[CatchesByPokeballWithHP] = []
    for pokemon in ["caterpie", "onix"]:
        for hp in HP_LEVELS:
            poke = factory.create(pokemon, 100, StatusEffect.NONE, hp.value)
            catches.append(catch_with_pokeball_with_hp(poke, 10_000, hp.value))

    print(catches)
    return catches



def ej1():
    catches: list[CatchesByPokeball] = []
    for pokemon in pokemons:
        poke = create_ideal_pokemon(factory, pokemon)
        catches.extend(catch_with_all_pokeballs(poke, 10_000))

    print(catches)

    df_1a, df_mean_1a = pandas_aggregate_1a(catches)

    print(df_1a)
    print(df_mean_1a)

    df_1b = pandas_aggregate_1b(catches)
    print(df_1b)

    plot_1a(df_mean_1a)
    plot_1b(df_1b)


def ej2():
    print("************ Ejercicio 2 ************")
    catches = ej2b()

    #df_2b, df_mean_2b = pandas_aggregate_2b(catches)

    #print(df_2b)
    #print(df_mean_2b)

    #df_2b = pandas_aggregate_2b(catches)
    #print(df_2b)

    #plot_1a(df_mean_2b)
    #plot_1b(df_2b)

def pandas_aggregate_2b(catches: list[CatchesByPokeballWithHP]):
    data = [
        {
            "pokemon": catch.pokemon.name,
            "hp": catch.hp,
            "catches": np.sum(catch.catches),
            "throws": len(catch.catches),
        }
        for catch in catches
    ]
    df = pd.DataFrame(data).sort_values(by=["pokemon", "hp"]).reset_index(drop=True)
    df["mean"] = np.divide(df["catches"], df["throws"])

    pokeball_by_pokemon = df[df["hp"] == "hp"].set_index("pokemon")
    """
    df["relative_to_pokeball"] = df.apply(
        lambda row: (
            row["mean"] / pokeball_by_pokemon.loc[row["pokemon"]]["mean"]
            if row["ball"] != "pokeball"
            else 1.0
        ),
        axis="columns",
    )
    """

    return df

if __name__ == "__main__":
    pokemons = get_pokemons()
    factory = PokemonFactory(POKEMONS_CONFIG)

    #ej1()
    ej2()
