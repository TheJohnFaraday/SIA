from exercises.ej1_and import ej1_and
from exercises.ej1_xor import ej1_xor
from exercises.ej2 import ej2 as ej2_implementation
from exercises.ej3_xor import xor as ej3_xor

from src.configuration import Configuration, read_configuration


def ej1(config: Configuration):
    and_expected = ej1_and(config)
    ej1_xor(and_expected, config)


def ej2(config: Configuration):
    ej2_implementation(config)


def ej3(config: Configuration):
    ej3_xor(config)


def main():
    config = read_configuration()
    #ej1(config)
    ej2(config)
    #ej3(config)


if __name__ == "__main__":
    main()
