from dataclasses import dataclass
from typing import Any


@dataclass
class NetworkOutput:
    expected: Any
    output: Any
    error: float
