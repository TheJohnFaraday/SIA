from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


class FinishMethod(Enum):
    TIME = "time"
    MAX_GENERATIONS = "max_generations"
    STRUCTURE = "structure"
    CONTENT = "content"
    ACCEPTABLE_FITNESS = "acceptable_fitness"


@dataclass(frozen=True)
class Configuration:
    methods: list[FinishMethod]
    time_limit: int = 0
    max_generations: int = 0
    structure: Any = None  # TODO
    content_generations: int = 0
    acceptable_fitness: int = 0


class Finish:
    def __init__(self, configuration: Configuration):
        self.__configuration = configuration

    def done(self) -> bool:
        raise NotImplementedError("Finish not implemented :)")
