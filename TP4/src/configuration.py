import tomllib
from dataclasses import dataclass


@dataclass(frozen=True)
class KohonenConfig:
    k: int
    initial_radius: float
    epochs_multiplier: int


def read_configuration():
    with open("../config.toml", "rb") as f:
        data = tomllib.load(f)

        k = int(data.get("kohonen", {}).get("k", 2))
        initial_radius = float(data.get("kohonen", {}).get("initial_radius", 1.414))
        epochs_multiplier = int(data.get("kohonen", {}).get("epochs_multiplier", 25))

        return KohonenConfig(k=k, initial_radius=initial_radius, epochs_multiplier=epochs_multiplier)
