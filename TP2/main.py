from src.configuration import read_configuration
from src.Selection import Selection


if __name__ == "__main__":
    configuration = read_configuration()
    selection = Selection(population_sample=configuration.population_sample, configuration=configuration.selection)
