import numpy as np
import matplotlib.pyplot as plt

from src.configuration import Configuration

from src.Dense import Dense
from src.activation_functions import Tanh
from src.errors import MSE

from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Optimizer import GradientDescent, Momentum, Adam
from src.Training import Online


def xor(config: Configuration):
    X = np.reshape(config.xor_input, (4, 2, 1))
    Y = np.reshape(config.xor_output, (4, 1, 1))

    network = [
        # Dense(2, 3, GradientDescent(config.learning_rate)),
        # Dense(2, 3, Momentum(config.learning_rate, config.multilayer.momentum)),
        config.multilayer.parity_discrimination_activation_function,
        Dense(
            2,
            1,
            Adam(
                learning_rate=config.learning_rate,
                beta1=config.multilayer.beta1,
                beta2=config.multilayer.beta2,
                epsilon=config.multilayer.epsilon,
            ),
        ),
        # Dense(3, 1, GradientDescent(config.learning_rate)),
        # Dense(3, 1, Momentum(config.learning_rate, config.multilayer.momentum)),
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

    for x, y in zip(X, Y):
        print(f"XOR Input: {x}")
        output = MultiLayerPerceptron.predict(new_network, x)
        print(f"XOR Expected Output: {y}")
        print(f"XOR Output: {output}")
