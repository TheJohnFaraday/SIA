import numpy as np
import matplotlib.pyplot as plt

from src.configuration import Configuration

from src.Dense import Dense
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent
from src.Training import Online


def is_odd(config: Configuration):
    is_odd_output = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X = np.reshape(config.multilayer.digits_input, (10, 35, 1))
    Y = np.reshape(is_odd_output, (10, 1, 1))
    network = [
        Dense(35, 70, GradientDescent(config.learning_rate)),
        config.multilayer.parity_discrimination_activation_function,
        Dense(70, 1, GradientDescent(config.learning_rate)),
        config.multilayer.parity_discrimination_activation_function,
    ]

    mlp = MultiLayerPerceptron(
        Online(MultiLayerPerceptron.predict),
        network,
        MSE(),
        config.epoch,
        config.learning_rate,
    )

    new_network = mlp.train(X, Y)

    for x, y in zip(X, is_odd_output):
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f"Is Odd Expected Output: {y}")
        print(f"Is Odd Output: {output}")


def which_number(config: Configuration):
    X = np.reshape(config.multilayer.digits_input, (10, 35, 1))
    Y = np.reshape(config.multilayer.digits_output, (10, 10, 1))
    network = [
        Dense(35, 10, GradientDescent(config.learning_rate)),
        config.multilayer.digits_discrimination_activation_function,
    ]

    mlp = MultiLayerPerceptron(
        Online(MultiLayerPerceptron.predict),
        network,
        MSE(),
        config.epoch,
        config.learning_rate,
    )

    new_network = mlp.train(X, Y)

    for x, y in zip(X, config.multilayer.digits_output):
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f"Number Expected Output: {y}")
        print(f"Number Output: {output / np.sqrt(np.sum(output**2))}")
