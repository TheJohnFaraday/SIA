import numpy as np
import errors

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def linear_activation(x):
    return x

class LinearPerceptron:
    def __init__(self, input_size, learning_rate, activation_function, error_function):
        # initialize weights w to small random values
        self.weights = np.random.rand(input_size) * 0.01
        # initialize bias to small random value
        self.bias = np.random.rand() * 0.01
        # set learning rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.error_function = error_function


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

    def test(self, data):
       #TODO probar si aprendio correctamente
       pass

if __name__ == "__main__":
    #TODO cambiar por datos csv y ver como se divide:
    linear_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    linear_expected = np.array([-1, -1, -1, 1]) 

    #Perceptron Lineal
    linear_perceptron = LinearPerceptron(2, 0.1, linear_activation, errors.MSE())
    linear_perceptron.train(linear_input, linear_expected, epochs=100)

    #TODO evaluar con los datos de prueba

    #Perceptron No Lineal
    #TODO cambiar por datos csv y ver como se divide:
    non_linear_input = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    non_linear_expected = np.array([-1, -1, -1, 1])

    non_linear_perceptron = LinearPerceptron(2, 0.1, sigmoid_activation, errors.MSE())
    non_linear_perceptron.train(non_linear_input, non_linear_expected, epochs=100)