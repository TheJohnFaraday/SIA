import numpy as np
import pandas as pd
from enum import Enum
from src.utils import unnormalize

from src.utils import unnormalize_0_1


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
    return 2 * beta * logistic_activation(x, beta) * (1 - logistic_activation(x, beta))


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
        error_function
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
                self.unnormalize = unnormalize
            case ActivationFunction.LOGISTIC:
                self.activation_function = logistic_activation
                self.activation_prime = logistic_prime
                self.unnormalize = unnormalize_0_1
            case ActivationFunction.LINEAR:
                self.activation_function = linear_activation
                self.activation_prime = linear_prime
                self.unnormalize = unnormalize
        self.beta = beta
        self.error_function = error_function
        self.final_error = []
        self.train_errors = []
        self.final_epochs = []

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        output = self.activation_function(weighted_sum, self.beta)
        prime = self.activation_prime(weighted_sum, self.beta)
        return (output, prime)

    def train(self, input, expected_output, epochs=100):
        total_error = 0
        for e in range(epochs):
            error = []
            self.final_epochs.append(e + 1)
            for x, y in zip(input, expected_output):
                # forward
                output, prime = self.predict(x)
                # error
                error.append(self.error_function.error(y, output))
                # backward
                self.weights += self.learning_rate * (y - output) * prime * x
                self.bias += self.learning_rate * (y - output) * prime
            total_error = 0.5 * np.sum(error)
            self.train_errors.append(total_error)
            # convergencia:
            if total_error < 0.01:
                break

        self.final_error.append(total_error)
        print(f"{e + 1}/{epochs}, error={total_error}")

    def test(self, input, expected_output, unnom_expected_output):
        predictions = []
        errors = []
        for x, y_true, unnom_y_true in zip(input, expected_output, unnom_expected_output):
            y_pred, _ = self.predict(x)
            predictions.append(y_pred)
            error = self.error_function.error(y_true, y_pred)
            errors.append(error)
            #print(f"Entrada: {x}, Predicción: {self.unnormalize(y_pred, np.min(unnom_expected_output), np.max(unnom_expected_output))}, "
            #      f"Valor Esperado: {self.unnormalize(y_true, np.min(unnom_expected_output), np.max(unnom_expected_output))}, Error: {error}")

        avg_error = np.mean(errors)
        print(f"Error medio en el conjunto de prueba: {avg_error}")
        return predictions, errors
