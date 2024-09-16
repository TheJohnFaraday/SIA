from time import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from functools import reduce
from .Player import Player


class FinishMethod(Enum):
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
    THRESHOLD = 5
    STRUCTURE_DELTA = Decimal("0.90")
    FITNESS_DELTA = Decimal("0.90")
    HEIGHT_MULTI = 20

    def __init__(self, configuration: Configuration):
        self.__configuration = configuration
        self.generation = 1
        self.threshold = 0
        self.ts_start = int(time())
        self.prior_p_structure = None
        self.prior_p_content = Decimal("0")

    def done(self, population: list[Player]) -> bool:
        if self.__eval(population):
            if self.threshold >= self.THRESHOLD:
                return True
        else:
            self.threshold = 0

        self.generation += 1
        self.prior_p_structure = self.compute_population_structure(population)
        self.prior_p_content = self.compute_population_content(population)
        return False

    def __eval(self, population: list[Player]):
        ts = int(time()) - self.ts_start
        if ts >= self.__configuration.time_limit:
            self.threshold += self.THRESHOLD
            return True
        for method in self.__configuration.methods:
            match method:
                case FinishMethod.MAX_GENERATIONS:
                    if self.generation >= self.__configuration.max_generations:
                        self.threshold += self.THRESHOLD
                        return True
                case FinishMethod.ACCEPTABLE_FITNESS:
                    if (
                        self.compute_population_content(population)
                        >= self.__configuration.acceptable_fitness
                    ):
                        self.threshold += self.THRESHOLD
                        return True
                case FinishMethod.STRUCTURE:
                    structure = self.compute_population_structure(population)
                    if (
                        self.compute_structure_delta(structure, self.prior_p_structure)
                        > self.STRUCTURE_DELTA
                    ):
                        self.threshold += 1
                        return True
                case FinishMethod.CONTENT:
                    content = self.compute_population_content(population)
                    delta = 0
                    if content < self.prior_p_content:
                        delta = content / self.prior_p_content
                    else:
                        delta = self.prior_p_content / content
                    if delta > self.FITNESS_DELTA:
                        self.threshold += 1
                        return True
        return False

    @staticmethod
    def compute_structure_delta(
        struct1: PopulationStructure, struct2: PopulationStructure
    ):
        delta = 0
        delta += min(struct2.height, struct1.height) / max(
            struct2.height, struct1.height
        )
        delta += Decimal(min(struct2.strength, struct1.strength)) / Decimal(
            max(struct2.strength, struct1.strength)
        )
        delta += Decimal(min(struct2.endurance, struct1.endurance)) / Decimal(
            max(struct2.endurance, struct1.endurance)
        )
        delta += Decimal(min(struct2.intelligence, struct1.intelligence)) / Decimal(
            max(struct2.intelligence, struct1.intelligence)
        )
        delta += Decimal(min(struct2.dexterity, struct1.dexterity)) / Decimal(
            max(struct2.dexterity, struct1.dexterity)
        )
        delta += Decimal(min(struct2.physique, struct1.physique)) / Decimal(
            max(struct2.physique, struct1.physique)
        )

        return Decimal(delta / 6)

    @staticmethod
    def compute_population_structure(population: list[Player]):
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
