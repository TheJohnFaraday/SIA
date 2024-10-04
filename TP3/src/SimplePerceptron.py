import numpy as np

def step_activation(x):
    # TODO chequear si el else es 0 o -1
    return 1 if x > 0 else -1

class SimplePerceptron:
    def __init__(self, input_size, learning_rate, activation_function):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size) * 0.01
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function


    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, input, expected_output, epochs=100):
        total_error = 0
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
            total_error += abs(error)
            #convergencia:
            #if total_error < 0.01:
             #   break


if __name__ == "__main__":
    #AND:
    and_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    and_expected = np.array([-1, -1, -1, 1])
    and_perceptron = SimplePerceptron(2, 0.1, step_activation)
    and_perceptron.train(and_input, and_expected, epochs=100)

    for x, y in zip(and_input, and_expected):
        pred = and_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")

    '''#XOR:
    xor_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    xor_expected = np.array([1, 1, -1, -1])
    xor_perceptron = SimplePerceptron(2, 0.1, step_activation)
    xor_perceptron.train(xor_input, xor_expected, epochs=100)

    for x, y in zip(xor_input, xor_expected):
        pred = xor_perceptron.predict(x)
        print(f"Entrada: {x}, Predicción: {pred}, Valor Esperado: {y}")'''
    
