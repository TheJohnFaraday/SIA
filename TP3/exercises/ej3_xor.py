import numpy as np
import matplotlib.pyplot as plt

from src.configuration import Configuration

from src.Dense import Dense
from src.activation_functions import Tanh
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent
from src.Training import Online

# reshape para hacerla de 2x1 (dense layer recibe columnas de input)
# X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
# Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))


def xor(config: Configuration):
    X = np.reshape(config.xor_input, (4, 2, 1))
    Y = np.reshape(config.xor_output, (4, 1, 1))

    network = [
        Dense(2, 3, GradientDescent(config.learning_rate)),
        Tanh(),
        Dense(3, 1, GradientDescent(config.learning_rate)),
        Tanh(),
    ]

    mlp = MultiLayerPerceptron(
        Online(MultiLayerPerceptron.predict),
        network,
        MSE(),
        config.epoch,
        config.learning_rate,
    )

    new_network = mlp.train(X, Y)

    for x, y in zip(X, Y):
        print(f"XOR Input: {x}")
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f'XOR Expected Output: {y}')
        print(f"XOR Output: {output}")
