import enum
import tomllib
from dataclasses import dataclass
from enum import Enum
from typing import Any
from src.SOM import DistanceType


class ConfigurationToRead(Enum):
    KOHONEN = enum.auto()
    OJA = enum.auto()


@dataclass(frozen=True)
class KohonenConfig:
    k: int
    initial_radius: float
    epochs_multiplier: int
    set_initial_weights_from_dataset: bool
    distance: DistanceType


@dataclass(frozen=True)
class OjaConfig:
    max_epochs: int
    learning_rate: float
    weights_low: float
    weights_high: float


def read_kohonen_configuration(data: dict[str, Any]):
    k = int(data.get("k", 2))
    initial_radius = float(data.get("initial_radius", 1.414))
    epochs_multiplier = int(data.get("epochs_multiplier", 25))
    set_initial_weights_from_dataset = bool(data.get("set_initial_weights_from_dataset", True))
    distance_str = data.get("distance", "euclidean")

    try:
        distance = DistanceType(distance_str)
    except ValueError:
        raise ValueError(f"Invalid distance type '{distance_str}' in config")

    return KohonenConfig(
        k=k, initial_radius=initial_radius, epochs_multiplier=epochs_multiplier,
        set_initial_weights_from_dataset=set_initial_weights_from_dataset, distance=distance
    )


def read_oja_configuration(data: dict[str, Any]):
    return OjaConfig(
        max_epochs=data["epochs"],
        learning_rate=data["learning_rate"],
        weights_low=data["weights_low"],
        weights_high=data["weights_high"],
    )


def read_configuration(configuration_to_read: ConfigurationToRead):
    with open("config.toml", "rb") as f:
        data = tomllib.load(f)

        match configuration_to_read:
            case ConfigurationToRead.KOHONEN:
                return read_kohonen_configuration(data["kohonen"])

            case ConfigurationToRead.OJA:
                return read_oja_configuration(data["oja"])
