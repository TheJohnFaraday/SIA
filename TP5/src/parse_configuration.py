import tomllib

from src.Configuration import Configuration, AdamConfiguration, NoiseConfiguration


def read_configuration(file_path: str):
    with open(file_path, "rb") as f:
        data = tomllib.load(f)

        raw_adam: dict | None = data.get("adam", None)
        adam: AdamConfiguration | None = None
        if raw_adam:
            adam = AdamConfiguration(
                beta1=raw_adam["beta1"],
                beta2=raw_adam["beta2"]
            )

        raw_noise: dict | None = data.get("noise", None)
        noise: NoiseConfiguration | None = None
        if raw_noise:
            noise = NoiseConfiguration(
                intensity=raw_noise["intensity"],
                spread=raw_noise["spread"]
            )


        return Configuration(
            plot=data["plot"],
            seed=data.get("seed", None),
            learning_rate=data["learning_rate"],
            beta=data["beta"],
            epsilon=data["epsilon"],
            batch_size=data["batch_size"],
            epochs=data["epochs"],
            adam=adam,
            noise=noise
        )
