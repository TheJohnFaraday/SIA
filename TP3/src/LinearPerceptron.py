import numpy as np
import pandas as pd
from enum import Enum


class ActivationFunction(Enum):
    TANH = "tanh"
    LOGISTIC = "logistic"
    LINEAR = "linear"


def tanh_activation(x, beta):
    return np.tanh(beta * x)


def tanh_prime(x, beta):
    return beta * (1 - tanh_activation(x, beta) ** 2)


def logistic_activation(x, beta):
    return 1 / (1 + np.exp(-2 * beta * x))


def logistic_prime(x, beta):
    2 * beta * logistic_activation(x, beta)(1 - logistic_activation(x, beta))


def linear_activation(x, beta):
    return x


def linear_prime(x, beta):
    return 1


class LinearPerceptron:
    def __init__(
        self,
        input_size,
        learning_rate,
        activation_function,
        beta,
        error_function,
    ):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size)
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate
        match activation_function:
            case ActivationFunction.TANH:
                self.activation_function = tanh_activation
                self.activation_prime = tanh_prime
            case ActivationFunction.LOGISTIC:
                self.activation_function = logistic_activation
                self.activation_prime = logistic_prime
            case ActivationFunction.LINEAR:
                self.activation_function = linear_activation
                self.activation_prime = linear_prime
        self.beta = beta
        self.error_function = error_function

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        output = self.activation_function(weighted_sum, self.beta)
        prime = self.activation_prime(weighted_sum, self.beta)
        return (output, prime)

    def train(self, input, expected_output, epochs=100):
        total_error = 0
        for e in range(epochs):
            error = []
            for x, y in zip(input, expected_output):
                # forward
                output, prime = self.predict(x)
                # error
                error.append(self.error_function.error(y, output))
                # backward
                self.weights += self.learning_rate * (y - output) * prime * x
                self.bias += self.learning_rate * (y - output) * prime
            total_error = 0.5 * np.sum(error)
            # convergencia:
            if total_error < 0.01:
                break
        print(f"{e + 1}/{epochs}, error={total_error}")

    def test(self, data):
        # TODO probar si aprendio correctamente
        pass
