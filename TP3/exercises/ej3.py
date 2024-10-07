import os
import numpy as np
import matplotlib.pyplot as plt

from src.configuration import Configuration

from src.Dense import Dense
from src.activation_functions import Tanh, Logistic
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent
from src.Training import Online
from src.utils import unnormalize


def is_odd(config: Configuration):
    is_odd_output = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X = np.reshape(config.digits_input, (10, 35, 1))
    Y = np.reshape(is_odd_output, (10, 1, 1))
    network = [
        Dense(35, 70, GradientDescent(config.learning_rate)),
        Logistic(config.beta),
        Dense(70, 1, GradientDescent(config.learning_rate)),
        Logistic(config.beta),
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
    X = np.reshape(config.digits_input, (10, 35, 1))
    Y = np.reshape(config.digits_output_norm, (10, 1, 1))
    network = [
        Dense(35, 70, GradientDescent(config.learning_rate)),
        Tanh(config.beta),
        Dense(70, 1, GradientDescent(config.learning_rate)),
        Tanh(config.beta),
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
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f"Is Odd Expected Output: {y}")
        print(f"Is Odd Output: {unnormalize(output, 0, 9)}")


def xor(config: Configuration):
    X = np.reshape(config.xor_input, (4, 2, 1))
    Y = np.reshape(config.xor_output, (4, 1, 1))

    network = [
        Dense(2, 3, GradientDescent(config.learning_rate)),
        Tanh(config.beta),
        Dense(3, 1, GradientDescent(config.learning_rate)),
        Tanh(config.beta),
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
        print(f"XOR Expected Output: {y}")
        print(f"XOR Output: {output}")
