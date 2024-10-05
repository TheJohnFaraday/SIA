import numpy as np

def step_activation(x):
    # TODO chequear si el else es 0 o -1
    return 1 if x > 0 else -1

class SimplePerceptron:
    def __init__(self, input_size, learning_rate):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size) * 0.01
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate
        self.activation_function = step_activation


    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, input, expected_output, epochs=100):
        total_error = 0
        for e in range(epochs):
            error = []
            for x, y in zip(input, expected_output):
                # forward
                output = self.predict(x)
                # error
                error.append(abs(y - output))
                # backward
                self.weights += self.learning_rate * (y - output) * x
                self.bias += self.learning_rate * (y - output)
            total_error = np.linalg.norm(error, 1)
            print(f"{e + 1}/{epochs}, error={total_error}")
            # convergencia:
            if total_error < 0.01:
                break


if __name__ == "__main__":
    #AND:
    and_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    and_expected = np.array([-1, -1, -1, 1])
    and_perceptron = SimplePerceptron(len(and_input[0]), 0.001)
    and_perceptron.train(and_input, and_expected, epochs=100)

    for x, y in zip(and_input, and_expected):
        pred = and_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")

    #XOR:
    xor_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    xor_expected = np.array([1, 1, -1, -1])
    xor_perceptron = SimplePerceptron(len(xor_input[0]), 0.001)
    xor_perceptron.train(xor_input, xor_expected, epochs=10000)

    for x, y in zip(xor_input, xor_expected):
        pred = xor_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")

