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


@dataclass
class CatchesByPokeball:
    pokemon: Pokemon
    ball: Pokeballs
    catches: list[int]


@dataclass(frozen=True, eq=True)
class PokeballMean:
    ball: Pokeballs
    mean: float


def create_ideal_pokemon(factory: PokemonFactory, pokemon: str) -> Pokemon:
    return factory.create(pokemon, 100, StatusEffect.NONE, 1)


def catch_with_all_pokeballs(pokemon: Pokemon, times: int) -> list[CatchesByPokeball]:
    catches: list[CatchesByPokeball] = list()
    for ball in Pokeballs:
        catches_with_ball: list[int] = []
        for _ in range(times):
            catches_with_ball.append(1 if attempt_catch(pokemon, ball.value)[0] else 0)

        catches.append(CatchesByPokeball(pokemon, ball, catches_with_ball))

    return list(catches)


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
            row["mean"] / pokeball_by_pokemon.loc[row['pokemon']]["mean"]
            if row["ball"] != "pokeball"
            else 1.0
        ),
        axis="columns",
    )

    return df


if __name__ == "__main__":
    pokemons = get_pokemons()
    factory = PokemonFactory(POKEMONS_CONFIG)

    catches: list[CatchesByPokeball] = []
    for pokemon in pokemons:
        poke = create_ideal_pokemon(factory, pokemon)
        catches.extend(catch_with_all_pokeballs(poke, 10000))

    df_1a, df_mean_1a = pandas_aggregate_1a(catches)

    print(df_1a)
    print(df_mean_1a)

    df_1b = pandas_aggregate_1b(catches)
    print(df_1b)
