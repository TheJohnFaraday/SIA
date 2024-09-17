import tomllib
from dataclasses import dataclass
from decimal import Decimal

from .Cross import CrossoverMethod, Configuration as CrossoverConfiguration
from .Finish import FinishMethod, Configuration as FinishConfiguration
from .Mutation import MutationMethod, Configuration as MutationConfiguration
from .PlayerClass import PlayerClass
from .Replacement import ReplacementMethod, Configuration as ReplacementConfiguration
from .Selection import (
    Configuration as SelectionConfiguration,
    SelectionMethods,
)
from .utils import key_from_enum_value, key_from_enum_value_with_fallback


@dataclass(frozen=True)
class GeneticConfiguration:
    crossover: CrossoverConfiguration
    mutation: MutationConfiguration
    replacement: ReplacementConfiguration


@dataclass(frozen=True)
class Configuration:
    plot: bool
    player: PlayerClass
    points: int
    initial_population: int
    population_sample: int
    genetic: GeneticConfiguration
    selection: SelectionConfiguration
    replacement: ReplacementConfiguration
    finish: FinishConfiguration
    random_seed: int | None


def read_configuration():
    with open("config.toml", "rb") as f:
        data = tomllib.load(f, parse_float=Decimal)

        genetic_mutation = key_from_enum_value_with_fallback(
            MutationMethod, data["genetic"]["mutation"], MutationMethod.SINGLE
        )
        is_uniform = data["genetic"]["parameters"]["mutation"].get("is_uniform", False)

        mutation_configuration = MutationConfiguration(
            mutation=genetic_mutation,
            pm=data["genetic"]["parameters"]["mutation"]["pm"],
            is_uniform=is_uniform,
            max_genes=(
                data["genetic"]["parameters"]["mutation"]["limited_multi"]
                if genetic_mutation == MutationMethod.MULTI
                else 1
            ),
            generational_increment=(
                data["genetic"]["parameters"]["mutation"]["generational_increment"]
                if not is_uniform
                else Decimal(0)
            ),
            lower_bound=data["genetic"]["parameters"]["mutation"].get(
                "lower_bound", Decimal("-0.2")
            ),
            higher_bound=data["genetic"]["parameters"]["mutation"].get(
                "higher_bound", Decimal("0.2")
            ),
        )

        crossover_configuration = CrossoverConfiguration(
            method=key_from_enum_value_with_fallback(
                CrossoverMethod, data["genetic"]["crossover"], CrossoverMethod.ONE_POINT
            ),
            pc=data["genetic"]["parameters"]["crossover"]["pc"],
            uniform_crossover_probability=data["genetic"]["parameters"][
                "crossover"
            ].get("p", Decimal(0.5)),
        )

        selection_methods = [
            key_from_enum_value(SelectionMethods, method)
            for method in data["selection"]["method"]
        ]
        selection_configuration = SelectionConfiguration(
            method=selection_methods,
            weight=[weight for weight in data["selection"]["weight"]],
            deterministic_tournament_individuals_to_select=(
                data["selection"]["parameters"]["deterministic_tournament"][
                    "individuals_to_select"
                ]
                if SelectionMethods.DETERMINISTIC_TOURNAMENT in selection_methods
                else 0
            ),
            probabilistic_tournament_threshold=(
                data["selection"]["parameters"]["probabilistic_tournament"]["threshold"]
                if SelectionMethods.PROBABILISTIC_TOURNAMENT in selection_methods
                else 0
            ),
            boltzmann_temperature=(
                data["selection"]["parameters"]["boltzmann"]["temperature"]
                if SelectionMethods.BOLTZMANN in selection_methods
                else 0
            ),
        )

        replacement_configuration = ReplacementConfiguration(
            method=[
                key_from_enum_value(ReplacementMethod, method)
                for method in data["replacement"]["method"]
            ],
            weight=[weight for weight in data["selection"]["weight"]],
            gap=(
                data["selection"]["parameters"]["generational_gap"]["gap"]
                if ReplacementMethod.GENERATIONAL_GAP in selection_methods
                else 0
            ),
        )

        finish_methods = [
            key_from_enum_value(FinishMethod, method)
            for method in data["finish"]["method"]
        ]
        finish_configuration = FinishConfiguration(
            methods=finish_methods,
            threshold=data["finish"].get("threshold", 10),
            time_limit=(data["finish"]["time"].get("limit", 10)),
            max_generations=(
                data["finish"]["max_generations"]["generations"]
                if FinishMethod.MAX_GENERATIONS in finish_methods
                else 0
            ),
            structure=(
               data["finish"]["structure"].get("delta", Decimal("0.9"))
               if FinishMethod.STRUCTURE in finish_methods
               else Decimal(0.9)
            ),
            content_generations=(
                data["finish"]["content"]["generations"]
                if FinishMethod.CONTENT in finish_methods
                else None
            ),
            acceptable_fitness=(
                data["finish"]["acceptable_fitness"]["fitness"]
                if FinishMethod.ACCEPTABLE_FITNESS in finish_methods
                else None
            ),
        )

        genetic_configuration = GeneticConfiguration(
            crossover=crossover_configuration,
            mutation=mutation_configuration,
            replacement=replacement_configuration,
        )

        configuration = Configuration(
            plot=bool(data["plot"]),
            player=key_from_enum_value_with_fallback(
                PlayerClass, data["general"]["character"], fallback=PlayerClass.WARRIOR
            ),
            points=int(data["general"]["points"]),
            initial_population=int(data["general"]["initial_population"]),
            population_sample=int(data["general"]["population_sample"]),
            genetic=genetic_configuration,
            selection=selection_configuration,
            replacement=replacement_configuration,
            finish=finish_configuration,
            random_seed=data.get("seed", None),
        )

        return configuration
