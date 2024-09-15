from enum import Enum
from typing import Type, Any, Optional


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
