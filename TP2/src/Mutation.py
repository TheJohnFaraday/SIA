from enum import Enum
from dataclasses import dataclass
from decimal import Decimal

from .Player import Player


class MutationMethod(Enum):
    SINGLE = "single"
    LIMITED_MULTI = "limited_multi"
    UNIFORM_MULTI = "uniform_multi"
    COMPLETE = "complete"


@dataclass
class Configuration:
    mutation: MutationMethod
    pm: Decimal
    generational_increment: Decimal = Decimal(0)  # For not-uniform mutation method
    max_genes: int = 1  # For limited_multi mutation method


class Mutation:
    def __init__(self, configuration: Configuration):
        self.__configuration = configuration

    def mutate(self, generation: int, population: list[Player]):
        pass