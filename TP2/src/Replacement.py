from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class ReplacementMethod(Enum):
    FILL_ALL = "fill_all"
    FILL_PARENT = "fill_parent"
    GENERATIONAL_GAP = "generational_gap"


@dataclass(frozen=True)
class Configuration:
    method: list[ReplacementMethod]
    weight: list[Decimal]
