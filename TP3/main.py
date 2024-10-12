import random
import numpy as np

from exercises.ej1_and import ej1_and
from exercises.ej1_xor import ej1_xor
from exercises.ej2 import ej2 as ej2_implementation
from exercises.ej3_xor import xor as ej3_xor
from exercises.ej3 import is_odd as ej3_a, which_number as ej3_b
from exercises.ej4 import mnist_digit_clasification as ej4

from src.configuration import Configuration, read_configuration


def ej1(config: Configuration):
    and_expected = ej1_and(config)
    ej1_xor(and_expected, config)


def ej2(config: Configuration):
    ej2_implementation(config)


def ej3(config: Configuration):
    # ej3_xor(config)
    # ej3_a(config)
    ej3_b(config)


def err_handler(type, flag):
    print("Floating point error (%s), with flag %s" % (type, flag))


def main():
    config = read_configuration()
    if config.random_seed:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    saved_handler = np.seterrcall(err_handler)
    save_err = np.seterr(all='call')
    # ej1(config)
    ej2(config)
    # ej3(config)
    # ej4(config)


if __name__ == "__main__":
    main()
