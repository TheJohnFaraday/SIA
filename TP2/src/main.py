import toml

with open("config.toml", "r") as f:
    config = toml.load(f)



if __name__ == "__main__":
    print(config)
    role = config["role"]
    crossover_method = config["crossover"]
    selection_methods = config["selections"]
    mutation_method = config["mutation"]
    print(role)
