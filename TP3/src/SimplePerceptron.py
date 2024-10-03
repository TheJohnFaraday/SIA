import numpy as np


class SimplePerceptron:
    def __init__(self, input_size, learning_rate):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size) * 0.01
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate

    def step_activation(x):
        # TODO chequear si el else es 0 o -1
        return 1 if x > 0 else -1

    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.step_activation(weighted_sum)

    def train(self, input, expected_output, epochs=100):
        for e in range(epochs):
            error = 0
            for x, y in zip(input, expected_output):
                # forward
                output = self.predict(x)
                # error
                error += y - output
                # backward
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error
            error /= len(input)
            print(f"{e + 1}/{epochs}, error={error}")

        # TODO ver convergencia
