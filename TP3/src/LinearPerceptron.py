import numpy as np
from enum import Enum


class ActivationFunction(Enum):
    TANH = "tanh"
    LOGISTIC = "logistic"
    LINEAR = "linear"


class LinearNonLinearPerceptron:
    def __init__(self, input_size, learning_rate, beta, error_function):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size)
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate
        self.beta = beta
        self.error_function = error_function
        self.final_error = []
        self.train_errors = []
        self.final_epochs = []

    def activation_function(self, x, beta):
        raise NotImplementedError("Should be override by child implementation")

    def activation_prime(self, x, beta):
        raise NotImplementedError("Should be override by child implementation")

    def normalize(self, x, min, max):
        raise NotImplementedError("Should be override by child implementation")

    def unnormalize(self, x, min, max):
        raise NotImplementedError("Should be override by child implementation")

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        output = self.activation_function(weighted_sum, self.beta)
        prime = self.activation_prime(weighted_sum, self.beta)
        return (output, prime)

    def train(self, input, expected_output, epochs=100):
        total_error = 0
        min = np.min(expected_output)
        max = np.max(expected_output)
        for e in range(epochs):
            error = []
            self.final_epochs.append(e + 1)
            for x, y in zip(input, expected_output):
                # forward
                output, prime = self.predict(x)
                # error
                error.append(
                    self.unnormalize(
                        self.error_function.error(self.normalize(y, min, max), output),
                        min,
                        max,
                    )
                )
                # backward
                self.weights += (
                    self.learning_rate
                    * (self.normalize(y, min, max) - output)
                    * prime
                    * x
                )
                self.bias += (
                    self.learning_rate * (self.normalize(y, min, max) - output) * prime
                )
            total_error = 0.5 * np.sum(error)
            self.train_errors.append(total_error)
            # convergencia:
            if total_error < 0.01:
                break

        self.final_error.append(total_error)
        print(f"{e + 1}/{epochs}, error={total_error}")

    def test(self, input, expected_output):
        predictions = []
        errors = []
        min = np.min(expected_output)
        max = np.max(expected_output)
        for x, y_true in zip(input, expected_output):
            y_pred, _ = self.predict(x)
            predictions.append(self.unnormalize(y_pred, min, max))
            error = self.unnormalize(
                self.error_function.error(self.normalize(y_true, min, max), y_pred),
                min,
                max,
            )
            errors.append(error)
            # print(f"Entrada: {x}, Predicción: {self.unnormalize(y_pred, np.min(unnom_expected_output), np.max(unnom_expected_output))}, "
            #      f"Valor Esperado: {self.unnormalize(y_true, np.min(unnom_expected_output), np.max(unnom_expected_output))}, Error: {error}")

        avg_error = np.mean(errors)
        print(f"Error medio en el conjunto de prueba: {avg_error}")
        return predictions, errors


class TanhPerceptron(LinearNonLinearPerceptron):
    def __init__(self, input_size, learning_rate, beta, error_function):
        super().__init__(input_size, learning_rate, beta, error_function)

    def activation_function(self, x, beta):
        return np.tanh(beta * x)

    def activation_prime(self, x, beta):
        return beta * (1 - self.activation_function(x, beta) ** 2)

    def normalize(self, x, min, max):
        return (2 * (x - min) / (max - min)) - 1

    def unnormalize(self, x, min, max):
        return ((x + 1) * (max - min) / 2) + min


class LogisticPerceptron(LinearNonLinearPerceptron):
    def __init__(self, input_size, learning_rate, beta, error_function):
        super().__init__(input_size, learning_rate, beta, error_function)

    def activation_function(self, x, beta):
        return 1 / (1 + np.exp(-2 * beta * x))

    def activation_prime(self, x, beta):
        return (
            2
            * beta
            * self.activation_function(x, beta)
            * (1 - self.activation_function(x, beta))
        )

    def normalize(self, x, min, max):
        return (x - min) / (max - min)

    def unnormalize(self, x, min, max):
        return x * (max - min) + min


class LinearPerceptron(LinearNonLinearPerceptron):
    def __init__(self, input_size, learning_rate, beta, error_function):
        super().__init__(input_size, learning_rate, beta, error_function)

    '''
    def train(self, input, expected_output, epochs=100):
        total_error = 0
        min = np.min(expected_output)
        max = np.max(expected_output)
        for e in range(epochs):
            error = []
            self.final_epochs.append(e + 1)
            for x, y in zip(input, expected_output):
                # forward
                output, prime = self.predict(x)
                # error
                error.append(
                    self.unnormalize(
                        self.error_function.error(
                            self.normalize(y, min, max),
                            self.normalize(output, min, max),
                        ),
                        min,
                        max,
                    )
                )
                # backward
                self.weights += (
                    self.learning_rate
                    * (self.normalize(y, min, max) - self.normalize(output, min, max))
                    * prime
                    * x
                )
                self.bias += (
                    self.learning_rate
                    * (self.normalize(y, min, max) - self.normalize(output, min, max))
                    * prime
                )
            total_error = 0.5 * np.sum(error)
            self.train_errors.append(total_error)
            # convergencia:
            if total_error < 0.01:
                break

        self.final_error.append(total_error)
        print(f"{e + 1}/{epochs}, error={total_error}")

    def test(self, input, expected_output):
        predictions = []
        errors = []
        min = np.min(expected_output)
        max = np.max(expected_output)
        for x, y_true in zip(input, expected_output):
            y_pred, _ = self.predict(x)
            predictions.append(y_pred)
            error = self.unnormalize(
                self.error_function.error(
                    self.normalize(y_true, min, max), self.normalize(y_pred, min, max)
                ),
                min,
                max,
            )
            errors.append(error)
            # print(f"Entrada: {x}, Predicción: {self.unnormalize(y_pred, np.min(unnom_expected_output), np.max(unnom_expected_output))}, "
            #      f"Valor Esperado: {self.unnormalize(y_true, np.min(unnom_expected_output), np.max(unnom_expected_output))}, Error: {error}")

        avg_error = np.mean(errors)
        print(f"Error medio en el conjunto de prueba: {avg_error}")
        return predictions, errors
    '''

    def activation_function(self, x, beta):
        return x

    def activation_prime(self, x, beta):
        return 1

    def normalize(self, x, min, max):
        return x

    def unnormalize(self, x, min, max):
        return x
