import tomllib

if __name__ == "__main__":
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

        print(config)
        role = config["role"]
        crossover_method = config["crossover"]
        selection_methods = config["selections"]
        mutation_method = config["mutation"]
        print(role)
