from enum import Enum


class MutationMethod(Enum):
    SINGLE = "single"
    LIMITED_MULTI = "limited_multi"
    UNIFORM_MULTI = "uniform_multi"
    COMPLETE = "complete"
