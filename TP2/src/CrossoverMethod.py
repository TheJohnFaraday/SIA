from enum import Enum


class CrossoverMethod(Enum):
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ANNULAR = "annular"
