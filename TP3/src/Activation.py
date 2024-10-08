from .Layer import Layer
import numpy as np


class Activation(Layer):
    # recibe la funcion de activacion y su derivada (activation_prime)
    def __init__(self, activation, activation_prime, beta):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        self.beta = beta

    def forward(self, input_matrix):
        print(f" ACTIVATION IDA----input_matrix: {input_matrix.shape}")
        self.input_matrix = input_matrix
        return self.activation(self.input_matrix, self.beta)

    def backward(self, output_gradient, learning_rate):
        print("-----activation-------")
        print(f"output gradient: {output_gradient.shape}")
        print(f"input matrix: {self.input_matrix.shape}")
        return np.multiply(
            output_gradient, self.activation_prime(self.input_matrix, self.beta)
        )
