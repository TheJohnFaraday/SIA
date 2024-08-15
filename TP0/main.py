import json

import numpy as np
import pandas as pd

from enum import Enum
from matplotlib import pyplot as plt
from dataclasses import dataclass

from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon

POKEMONS_CONFIG = "pokemon.json"

CUSTOM_PALETTE = ["#508fbe", "#f37120", "#4baf4e", "#f2cb31", "#c62d2a", "#9b27b0"]


class Pokeballs(Enum):
    POKEBALL = "pokeball"
    ULTRA_BALL = "ultraball"
    HEAVY_BALL = "heavyball"
    FAST_BALL = "fastball"


class HP_LEVELS(Enum):
    HP_100 = 1.00
    HP_95 = 0.95
    HP_90 = 0.90
    HP_85 = 0.85
    HP_80 = 0.80
    HP_75 = 0.75
    HP_70 = 0.70
    HP_65 = 0.65
    HP_60 = 0.60
    HP_55 = 0.55
    HP_50 = 0.50
    HP_45 = 0.45
    HP_40 = 0.40
    HP_35 = 0.35
    HP_30 = 0.30
    HP_25 = 0.25
    HP_20 = 0.2
    HP_15 = 0.15
    HP_10 = 0.10
    HP_05 = 0.05


@dataclass
class CatchesByPokeball:
    pokemon: Pokemon
    ball: Pokeballs
    catches: list[int]


@dataclass
class CatchesByLevel:
    pokemon: Pokemon
    level: int
    catches: list[int]


@dataclass
class CatchesByPokeballWithHP(CatchesByPokeball):
    hp: float


@dataclass
class CatchesByPokeballWithStatusEffect(CatchesByPokeball):
    status_effect: StatusEffect


@dataclass(frozen=True, eq=True)
class PokeballMean:
    ball: Pokeballs
    mean: float


def create_ideal_pokemon(factory: PokemonFactory, pokemon: str) -> Pokemon:
    return factory.create(pokemon, 100, StatusEffect.NONE, 1)


def catch_with_pokeball(
    pokemon: Pokemon, ball: Pokeballs, times: int
) -> CatchesByPokeball:
    catches_with_ball: list[int] = []
    for _ in range(times):
        catches_with_ball.append(1 if attempt_catch(pokemon, ball.value)[0] else 0)

    return CatchesByPokeball(pokemon, ball, catches_with_ball)


def catch_with_all_pokeballs(pokemon: Pokemon, times: int) -> list[CatchesByPokeball]:
    catches: list[CatchesByPokeball] = []
    for ball in Pokeballs:
        catches.append(catch_with_pokeball(pokemon, ball, times))

    return catches


def catch_with_pokeball_with_hp(
    pokemon: Pokemon, times: int, hp: float
) -> list[CatchesByPokeballWithHP]:
    catch = catch_with_pokeball(pokemon, Pokeballs.POKEBALL, times)
    extended_catch = CatchesByPokeballWithHP(
        catch.pokemon, catch.ball, catch.catches, hp
    )

    return extended_catch


def catch_with_pokeball_with_status_effect(
    pokemon: Pokemon, times: int, status_effect: StatusEffect
) -> [CatchesByPokeballWithStatusEffect]:
    catch = catch_with_pokeball(pokemon, Pokeballs.POKEBALL, times)
    extended_catch = CatchesByPokeballWithStatusEffect(
        catch.pokemon, catch.ball, catch.catches, status_effect
    )

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

    label_x = list(map(lambda label: str(label).title(), df.index.values))

    ax.bar(label_x, df["mean"], color=CUSTOM_PALETTE[: len(df)])

    ax.set_ylabel("Probabilidad de captura promedio")
    ax.set_xlabel("Pokeball")
    ax.set_title("Probabilidad de captura promedio por Pokeball")

    plt.savefig("plots/prob_por_pokeball.png")


def plot_1b(df: pd.DataFrame):
    fig, ax = plt.subplots(layout="constrained")

    pokemons = list(
        map(lambda label: str(label).title(), df["pokemon"].unique().tolist())
    )

    x = np.arange(len(pokemons))
    bar_width = 0.20
    multiplier = 0

    values = {}
    for ball in df["ball"].unique():
        values[ball] = tuple(df[df["ball"] == ball]["relative_to_pokeball"])

    for ball, means in values.items():
        offset = bar_width * multiplier
        ax.bar(x + offset, means, bar_width, label=ball.title())
        multiplier += 1

    ax.set_ylabel("Efectividad de captura relativa")
    ax.set_xlabel("Pokeballs agrupadas por Pokemon")
    ax.set_title("Efectividad de captura de cada Pokemon relativa a Pokeball")

    ax.set_xticks(x + bar_width, pokemons)
    ax.legend(loc="upper right", ncols=2)

    plt.savefig("plots/efectividad_por_pokeball.png")


def plot_2a(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.15
    x_positions = np.arange(len(df["status_effect"].unique()))

    df_none = df[df["status_effect"] == "NONE"].set_index("pokemon")["mean"]

    for i, pokemon in enumerate(df["pokemon"].unique()):
        df_pokemon = df[df["pokemon"] == pokemon].set_index("status_effect")

        relative_means = df_pokemon["mean"] / df_none.get(pokemon, 1)

        ax.bar(
            x_positions + i * bar_width,
            relative_means,
            color=CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)],
            width=bar_width,
            label=pokemon.title(),
        )

    ax.set_ylabel("Efectividad de captura relativa")
    ax.set_xlabel("Efecto de Estado")
    ax.set_title(
        "Efectividad de captura relativa por Efecto de Estado", fontsize=12, pad=20
    )

    ax.set_xticks(x_positions + bar_width * (len(df["pokemon"].unique()) - 1) / 2)

    label_x = list(
        map(lambda label: str(label).title(), df["status_effect"].unique().tolist())
    )
    ax.set_xticklabels(label_x, rotation=0)

    ax.set_xlim(
        [
            min(x_positions) - bar_width,
            max(x_positions) + bar_width * len(df["pokemon"].unique()),
        ]
    )

    ax.legend(title="Pokemon", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("plots/efectividad_por_estado_por_pokemon.png")


def plot_2a2(df: pd.DataFrame):
    fig, ax = plt.subplots()

    label_x = list(map(lambda label: str(label).title(), df.index.values))

    ax.bar(label_x, df["mean"], color=CUSTOM_PALETTE[: len(CUSTOM_PALETTE)])

    ax.set_ylabel("Efectividad de captura promedio relativa")
    ax.set_xlabel("Efecto de Estado")
    ax.set_title("Efectividad de captura promedio relativa según Efecto de Estado")

    plt.savefig("plots/prob_relativa_por_estado.png")


def plot_2b(df: pd.DataFrame, pokemon_name: str):
    plt.figure(figsize=(20, 6))
    fig, ax = plt.subplots()

    ax.scatter(df["hp"], df["mean"], color=CUSTOM_PALETTE[0], s=50)
    hp_values = [level.value for level in HP_LEVELS]
    ax.set_xticks(hp_values)
    ax.set_xticklabels([int((level.value * 100)) for level in HP_LEVELS])
    ax.tick_params(labelsize=8)

    ax.set_ylabel("Probabilidad de captura promedio")
    ax.set_xlabel("HP")
    ax.set_title(
        f"Probabilidad de captura promedio por HP - {pokemon_name.capitalize()}"
    )

    plt.savefig(f"plots/prob_por_hp_{pokemon_name}.png")


def plot_2c(df: pd.DataFrame, pokemon_name: str):
    fig, ax = plt.subplots()

    ax.scatter(df["level"], df["mean"], color=CUSTOM_PALETTE[2], s=50)
    ax.set_ylim(bottom=0, top=df["mean"].max() * 1.1)
    ax.set_ylabel("Probabilidad de captura promedio")
    ax.set_xlabel("Nivel")
    ax.set_title(
        f"Probabilidad de captura promedio por Nivel - {pokemon_name.capitalize()}"
    )

    plt.savefig(f"plots/prob_por_lvl_{pokemon_name.capitalize()}.png")


def ej2a():
    catches: list[CatchesByPokeballWithStatusEffect] = []
    for pokemon in pokemons:
        for status_effect in StatusEffect:
            poke = factory.create(pokemon, 100, status_effect, 1)
            catches.append(
                catch_with_pokeball_with_status_effect(poke, 10_000, status_effect)
            )

    # print(catches)
    return catches


def ej2b():
    catches: list[CatchesByPokeballWithHP] = []
    for pokemon in ["caterpie", "onix"]:
        for hp in HP_LEVELS:
            poke = factory.create(pokemon, 100, StatusEffect.NONE, hp.value)
            catches.append(catch_with_pokeball_with_hp(poke, 10_000, hp.value))

    # print(catches)
    return catches


def ej2c():
    catches: list[CatchesByLevel] = []
    for pokemon in ["jolteon", "caterpie"]:
        for lvl in range(10):
            poke = factory.create(pokemon, lvl * 10, StatusEffect.NONE, 1)
            catch: list[int] = []
            for _ in range(10_000):
                catch.append(
                    1
                    if attempt_catch(
                        poke, ("fastball" if pokemon == "jolteon" else "pokeball")
                    )[0]
                    else 0
                )
            catches.append(CatchesByLevel(pokemon, lvl * 10, catch))

    return catches


def ej1():
    print("************ Ejercicio 1 ************")
    catches: list[CatchesByPokeball] = []
    for pokemon in pokemons:
        poke = create_ideal_pokemon(factory, pokemon)
        catches.extend(catch_with_all_pokeballs(poke, 10_000))

    # print(catches)

    df_1a, df_mean_1a = pandas_aggregate_1a(catches)

    table = pd.crosstab(
        df_1a["pokemon"],
        df_1a["ball"],
        values=df_1a["catches"],
        aggfunc="sum",
    )
    print(table)

    # print(df_1a)
    # print(df_mean_1a)

    df_1b = pandas_aggregate_1b(catches)
    # print(df_1b)

    plot_1a(df_mean_1a)
    plot_1b(df_1b)


def ej2():
    print("************ Ejercicio 2 ************")

    catches = ej2a()

    df_2a, df_mean_2a = pandas_aggregate_2a(catches)
    # print(df_2a)

    plot_2a(df_2a)
    plot_2a2(df_mean_2a)

    catches = ej2b()

    df_2b = pandas_aggregate_2b(catches)

    df_2b_caterpie = df_2b[df_2b["pokemon"] == "caterpie"]
    df_2b_onix = df_2b[df_2b["pokemon"] == "onix"]

    # print(df_2b_onix)
    # print(df_2b_caterpie)

    plot_2b(df_2b_caterpie, "caterpie")
    plot_2b(df_2b_onix, "onix")

    catches = ej2c()

    df_2c = pandas_aggregate_2c(catches)

    df_2c_caterpie = df_2c[df_2c["pokemon"] == "caterpie"]
    df_2c_jolteon = df_2c[df_2c["pokemon"] == "jolteon"]

    plot_2c(df_2c_caterpie, "caterpie")
    plot_2c(df_2c_jolteon, "jolteon")


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

    return df


def pandas_aggregate_2c(catches: list[CatchesByLevel]):
    data = [
        {
            "pokemon": catch.pokemon,
            "level": catch.level,
            "catches": np.sum(catch.catches),
            "throws": len(catch.catches),
        }
        for catch in catches
    ]
    df = pd.DataFrame(data).sort_values(by=["pokemon", "level"]).reset_index(drop=True)
    df["mean"] = np.divide(df["catches"], df["throws"])
    # print(df["mean"])

    return df


def pandas_aggregate_2a(catches: list[CatchesByPokeballWithStatusEffect]):
    data = [
        {
            "pokemon": catch.pokemon.name,
            "status_effect": catch.status_effect.name,
            "catches": np.sum(catch.catches),
            "throws": len(catch.catches),
        }
        for catch in catches
    ]
    df = (
        pd.DataFrame(data)
        .sort_values(by=["pokemon", "status_effect"])
        .reset_index(drop=True)
    )
    df["mean"] = np.divide(df["catches"], df["throws"])

    grouped = df.groupby("status_effect").agg(
        catches=("catches", "sum"), throws=("throws", "sum")
    )
    grouped["mean"] = grouped["catches"] / grouped["throws"]
    grouped = grouped.sort_values(by=["status_effect"])

    return df, grouped


def configure_matplotlib():
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CUSTOM_PALETTE)


if __name__ == "__main__":
    configure_matplotlib()
    pokemons = get_pokemons()
    factory = PokemonFactory(POKEMONS_CONFIG)

    ej1()
    ej2()
