import tomllib
from dataclasses import dataclass
from decimal import Decimal

from src.Cross import CrossoverMethod
from src.Finish import FinishMethod, Configuration as FinishConfiguration
from src.Mutation import MutationMethod
from src.PlayerClass import PlayerClass
from src.Replacement import ReplacementMethod, Configuration as ReplacementConfiguration
from src.Selection import (
    Configuration as SelectionConfiguration,
    SelectionMethods,
)
from src.utils import key_from_enum_value, key_from_enum_value_with_fallback


@dataclass(frozen=True)
class GeneticConfiguration:
    crossover: CrossoverMethod
    mutation: MutationMethod
    pm: Decimal
    max_genes: int = 0  # For limited_multi mutation method


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


if __name__ == "__main__":
    configuration: Configuration | None = None
    with open("config.toml", "rb") as f:
        data = tomllib.load(f, parse_float=Decimal)

        genetic_mutation = key_from_enum_value_with_fallback(
            MutationMethod, data["genetic"]["mutation"], MutationMethod.SINGLE
        )
        genetic_configuration = GeneticConfiguration(
            crossover=key_from_enum_value_with_fallback(
                CrossoverMethod, data["genetic"]["crossover"], CrossoverMethod.ONE_POINT
            ),
            mutation=genetic_mutation,
            pm=data["genetic"]["parameters"]["mutation"]["pm"],
            max_genes=(
                data["genetic"]["parameters"]["mutation"]["limited_multi"]
                if genetic_mutation == MutationMethod.LIMITED_MULTI
                else 0
            ),
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
        )

        finish_methods = [
            key_from_enum_value(FinishMethod, method)
            for method in data["finish"]["method"]
        ]
        finish_configuration = FinishConfiguration(
            methods=finish_methods,
            time_limit=(
                data["finish"]["time"]["limit"]
                if FinishMethod.TIME in finish_methods
                else 0
            ),
            max_generations=(
                data["finish"]["max_generations"]["generations"]
                if FinishMethod.MAX_GENERATIONS in finish_methods
                else 0
            ),
            structure=(
                data["finish"]["structure"]["structure"]
                if FinishMethod.STRUCTURE in finish_methods
                else None
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
        )

    print(configuration)
