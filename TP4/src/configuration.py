import enum
import tomllib
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ConfigurationToRead(Enum):
    KOHONEN = enum.auto()
    OJA = enum.auto()


@dataclass(frozen=True)
class KohonenConfig:
    k: int
    initial_radius: float
    epochs_multiplier: int


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

    return KohonenConfig(
        k=k, initial_radius=initial_radius, epochs_multiplier=epochs_multiplier
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
