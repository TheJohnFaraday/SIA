from time import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from functools import reduce
from .Player import Player


class FinishReason(Enum):
    MAX_GENERATIONS = "Maximum generation reached"
    STRUCTURE = "Structure has not been changing that much"
    CONTENT = "Content has not been changing that much"
    ACCEPTABLE_FITNESS = "Acceptable Fitness reached 💪"
    TIME_LIMIT = "Time limit exceeded"


class FinishMethod(Enum):
    MAX_GENERATIONS = "max_generations"
    STRUCTURE = "structure"
    CONTENT = "content"
    ACCEPTABLE_FITNESS = "acceptable_fitness"


@dataclass(frozen=True)
class Configuration:
    methods: list[FinishMethod]
    threshold: int
    structure: Decimal
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
    FITNESS_DELTA = Decimal("0.90")
    HEIGHT_MULTI = 20

    def __init__(self, configuration: Configuration):
        self.__configuration = configuration
        self.generation = 0
        self.threshold = 0
        self.ts_start = int(time())
        self.prior_p_structure = None
        self.prior_p_content = Decimal("0")
        self.finish_reason: FinishReason | None = None

    def done(self, population: list[Player], generation: int) -> bool:
        if self.__eval(population):
            if self.threshold >= self.__configuration.threshold:
                return True
        else:
            self.threshold = 0

        self.generation = generation
        self.prior_p_structure = self.compute_population_structure(population)
        self.prior_p_content = self.compute_population_content(population)
        return False

    @property
    def reason(self):
        if not self.finish_reason:
            raise RuntimeError("WE HAVE NOT FINISHED! (Now we did)")
        return self.finish_reason.value

    def __eval(self, population: list[Player]):
        ts = int(time()) - self.ts_start
        if ts >= self.__configuration.time_limit:
            self.threshold += self.__configuration.threshold
            self.finish_reason = FinishReason.TIME_LIMIT
            return True

        for method in self.__configuration.methods:
            match method:
                case FinishMethod.MAX_GENERATIONS:
                    if self.generation >= self.__configuration.max_generations:
                        self.threshold += self.__configuration.threshold
                        self.finish_reason = FinishReason.MAX_GENERATIONS
                        return True

                case FinishMethod.ACCEPTABLE_FITNESS:
                    if (
                        self.compute_population_content(population)
                        >= self.__configuration.acceptable_fitness
                    ):
                        self.threshold += self.__configuration.threshold
                        self.finish_reason = FinishReason.ACCEPTABLE_FITNESS
                        return True
                case FinishMethod.STRUCTURE:
                    structure = self.compute_population_structure(population)
                    if not self.prior_p_structure:
                        self.prior_p_structure = structure

                    if (
                        self.compute_structure_delta(structure, self.prior_p_structure)
                        > self.__configuration.structure
                    ):
                        self.threshold += 1
                        if self.threshold >= self.__configuration.threshold:
                            self.finish_reason = FinishReason.STRUCTURE
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
                        if self.threshold >= self.__configuration.threshold:
                            self.finish_reason = FinishReason.CONTENT
                        return True
        return False

    @staticmethod
    def compute_structure_delta(
        struct1: PopulationStructure, struct2: PopulationStructure
    ):
        total_deltas = 0
        delta = 0

        if struct1.height > 0 or struct1.height > 0:
            total_deltas += 1
            delta += min(struct2.height, struct1.height) / max(
                struct2.height, struct1.height
            )

        if struct1.strength > 0 or struct1.strength > 0:
            total_deltas += 1
            delta += Decimal(min(struct2.strength, struct1.strength)) / Decimal(
                max(struct2.strength, struct1.strength)
            )

        if struct1.endurance > 0 or struct1.endurance > 0:
            total_deltas += 1
            delta += Decimal(min(struct2.endurance, struct1.endurance)) / Decimal(
                max(struct2.endurance, struct1.endurance)
            )

        if struct1.intelligence > 0 or struct1.intelligence > 0:
            total_deltas += 1
            delta += Decimal(min(struct2.intelligence, struct1.intelligence)) / Decimal(
                max(struct2.intelligence, struct1.intelligence)
            )

        if struct1.dexterity > 0 or struct1.dexterity > 0:
            total_deltas += 1
            delta += Decimal(min(struct2.dexterity, struct1.dexterity)) / Decimal(
                max(struct2.dexterity, struct1.dexterity)
            )

        if struct1.physique > 0 or struct1.physique > 0:
            total_deltas += 1
            delta += Decimal(min(struct2.physique, struct1.physique)) / Decimal(
                max(struct2.physique, struct1.physique)
            )

        if total_deltas == 0:
            return 1

        return Decimal(delta / total_deltas)

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
