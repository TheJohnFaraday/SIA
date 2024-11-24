import tomllib

from src.Configuration import Configuration


def read_configuration(file_path: str):
    with open(file_path, "rb") as f:
        data = tomllib.load(f)

        return Configuration(
            plot=data["plot"],
            seed=data.get("seed", None),
            learning_rate=data["learning_rate"],
            beta=data["beta"],
            epsilon=data["epsilon"],
            batch_size=data["batch_size"],
            epochs=data["epochs"]
        )
