import json
import sys

import numpy as np

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
    catches: list[bool]


@dataclass(frozen=True, eq=True)
class PokeballMean:
    ball: Pokeballs
    mean: float


def create_ideal_pokemon(factory: PokemonFactory, pokemon: str) -> Pokemon:
    return factory.create(pokemon, 100, StatusEffect.NONE, 1)


def catch_with_all_pokeballs(pokemon: Pokemon, times: int) -> list[CatchesByPokeball]:
    catches: list[CatchesByPokeball] = list()
    for ball in Pokeballs:
        catches_with_ball: list[bool] = []
        for _ in range(times):
            catches_with_ball.append(attempt_catch(pokemon, ball.value)[0])

        catches.append(CatchesByPokeball(pokemon, ball, catches_with_ball))

    return list(catches)


def get_pokemons():
    pokemons: set[str] = set()
    with open(POKEMONS_CONFIG) as f:
        pokemon_json = json.load(f)
        for name, _ in pokemon_json.items():
            pokemons.add(name)

    return list(pokemons)


def aggregate_by_pokeball(catches: list[CatchesByPokeball]) -> list[PokeballMean]:
    means: list[PokeballMean] = []
    for ball in Pokeballs:
        pokeball_catches = filter(lambda x: x.ball == ball, catches)
        flatten_pokeball_catches = [
            catch for c in pokeball_catches for catch in c.catches
        ]
        means.append(PokeballMean(ball, np.mean(flatten_pokeball_catches)))

    return means


if __name__ == "__main__":
    pokemons = get_pokemons()
    factory = PokemonFactory(POKEMONS_CONFIG)

    catches: list[CatchesByPokeball] = []
    for pokemon in pokemons:
        poke = create_ideal_pokemon(factory, pokemon)
        catches.extend(catch_with_all_pokeballs(poke, 100))

    means = aggregate_by_pokeball(catches)
    print(means)

    # with open(f"{sys.argv[1]}", "r") as f:
    #     config = json.load(f)
    #     ball = config["pokeball"]
    #     pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)

#         # for i in range(100, 1, -1):
#         #     pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, i / 100)
#         #     print(pokemon.current_hp)

#         print("No noise: ", attempt_catch(pokemon, ball))
#         for _ in range(10):
#             print("Noisy: ", attempt_catch(pokemon, ball, 0.15))
