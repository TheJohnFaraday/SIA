from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AdamConfiguration:
    beta1: float
    beta2: float


@dataclass(frozen=True)
class Configuration:
    plot: bool
    seed: Optional[int]
    learning_rate: float
    beta: float
    epsilon: float
    batch_size: int
    epochs: int
    adam: Optional[AdamConfiguration]
