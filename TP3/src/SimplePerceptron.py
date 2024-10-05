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
