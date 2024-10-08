from enum import Enum
import numpy as np
from typing import Type, Any, Optional


def unnormalize(y, min, max):
    return ((y + 1) * (max - min) / 2) + min


def normalize(y, min, max):
    return (2 * (y - min) / (max - min)) - 1


def normalize_0_1(y, min, max):
    return (y - min) / (max - min)


def unnormalize_0_1(y, min, max):
    return y * (max - min) + min


def normalize2(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def unnormalize2(y, min, max):
    return y*(max - min) + min


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
