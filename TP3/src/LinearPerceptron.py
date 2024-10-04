import numpy as np
import pandas as pd
import errors


def tanh_activation(x, beta):
    return np.tanh(beta * x)


def tanh_prime(x, beta):
    return beta * (1 - np.tanh(x) ** 2)


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
        activation_prime,
        beta,
        error_function,
    ):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size) * 0.01
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_prime = activation_prime
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


if __name__ == "__main__":
    # TODO cambiar por datos csv y ver como se divide:
    df = pd.read_csv("TP3/datasets/TP3-ej2-conjunto.csv", header=0)
    input = np.array(
        list(
            map(lambda row: [row[1]["x1"], row[1]["x2"], row[1]["x3"]], df.iterrows())
        )  # (index, row)
    )
    expected = np.array(list(map(lambda row: row[1]["y"], df.iterrows())))

    # Perceptron Lineal
    linear_perceptron = LinearPerceptron(
        len(input[0]), 0.0001, linear_activation, linear_prime, 0.2, errors.MSE()
    )
    linear_perceptron.train(input, expected, epochs=10000)

    # TODO evaluar con los datos de prueba
    for x, y in zip(input, expected):
        pred, _ = linear_perceptron.predict(x)
        print(f"Lineal: Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")

    non_linear_perceptron = LinearPerceptron(
        len(input[0]), 0.0001, tanh_activation, tanh_prime, 0.2, errors.MSE()
    )
    non_linear_perceptron.train(input, expected, epochs=10000)

    for x, y in zip(input, expected):
        pred, _ = non_linear_perceptron.predict(x)
        print(f"No Lineal: Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")
