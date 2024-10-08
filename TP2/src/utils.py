import random
from enum import Enum
from typing import Type, Any, Optional


def random_numbers_that_sum_n(numbers_count: int, n: int) -> list[int]:
    """Returns a list of random numbers and `numbers_count` length whose elements sum `n`"""
    points = [0] + sorted(random.sample(range(1, n), numbers_count - 1)) + [n]
    return [points[i + 1] - points[i] for i in range(len(points) - 1)]


def key_from_enum_value(enum: Type[Enum], value: Any) -> Optional[Enum]:
    """
    Returns the first key in `enum` that has `value` as its value. If no key is found, returns `fallback`.
    """
    return key_from_enum_value_with_fallback(enum, value, fallback=None)


def key_from_enum_value_with_fallback(
    enum: Type[Enum], value: Any, fallback: Enum
) -> Enum:
    """
    Returns the first key in `enum` that has `value` as its value. If no key is found, returns `fallback`.
    """
    return next(filter(lambda enum_type: enum_type.value == value, enum), fallback)
