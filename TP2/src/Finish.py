from time import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any
from functools import reduce
from .Player import Player


class FinishMethod(Enum):
    TIME = "time"
    MAX_GENERATIONS = "max_generations"
    STRUCTURE = "structure"
    CONTENT = "content"
    ACCEPTABLE_FITNESS = "acceptable_fitness"


@dataclass(frozen=True)
class Configuration:
    methods: list[FinishMethod]
    time_limit: int = 10  # Seconds allowed for the algorithm to work
    max_generations: int = 50
    content_generations: Decimal = Decimal("0")
    acceptable_fitness: Decimal = Decimal("0")


@dataclass(eq=True, frozen=True)
class PopulationStructure:
    height: Decimal
    strength: int
    dexterity: int
    intelligence: int
    endurance: int
    physique: int


class Finish:
    def __init__(self, configuration: Configuration):
        self.__configuration = configuration
        self.generation = 1
        self.threshold = 0
        self.ts_start = int(time())
        self.prior_p_structure = None
        self.prior_p_content = Decimal("0")

    def done(self, population: list[Player]) -> bool:
        if self.__eval(population) and self.threshold >= 5:
            return True

        self.generation += 1
        self.prior_p_structure = self.compute_population_structure(population)
        self.prior_p_content = self.compute_population_content(population)
        return False

    def __eval(self, population: list[Player]):
        ts = int(time()) - self.ts_start
        if ts >= self.configuration.time_limit:
            self.threshold += 5
            return True
        for method in self.configuration.methods:
            match method:
                case FinishMethod.MAX_GENERATIONS:
                    if self.generation >= self.configuration.max_generations:
                        self.threshold += 5
                        return True
                case FinishMethod.ACCEPTABLE_FITNESS:
                    if (
                        self.compute_population_content(population)
                        >= self.configuration.acceptable_fitness
                    ):
                        self.threshold += 5
                        return True
                case FinishMethod.STRUCTURE:
                    structure = self.compute_population_structure(population)
                    if (
                        self.compute_structure_delta(structure, self.prior_p_structure)
                        < 40
                    ):
                        self.threshold += 1
                        return True
                case FinishMethod.CONTENT:
                    content = self.compute_population_content(population)
                    if content - self.prior_p_content < Decimal("1.5"):
                        self.threshold += 1
                        return True
        return False

    @staticmethod
    def compute_structure_delta(
        struct1: PopulationStructure, struct2: PopulationStructure
    ) -> int:
        delta = 0
        delta += abs(int(struct1.height-struct2.height)*20)
        delta += abs(struct1.strength - struct2.strength)
        delta += abs(struct1.endurance - struct2.endurance)
        delta += abs(struct1.intelligence - struct2.intelligence)
        delta += abs(struct1.dexterity - struct2.dexterity)
        delta += abs(struct1.physique - struct2.physique)
        return delta

    @staticmethod
    def compute_population_structure(population: list[Player]) -> Decimal:
        struct = [Decimal("0"), 0, 0, 0, 0, 0]
        for elem in population:
            struct[0] = elem.height
            struct[1] = elem.p_attr.strength
            struct[2] = elem.p_attr.dexterity
            struct[3] = elem.p_attr.intelligence
            struct[4] = elem.p_attr.endurance
            struct[5] = elem.p_attr.physique
        return PopulationStructure(
            height=struct[0],
            strength=struct[1],
            dexterity=struct[2],
            intelligence=struct[3],
            endurance=struct[4],
            physique=struct[5],
        )

    @staticmethod
    def compute_population_content(population: list[Player]):
        return reduce(lambda x, elem: x + elem.fitness, population, Decimal("0")) / len(
            population
        )
