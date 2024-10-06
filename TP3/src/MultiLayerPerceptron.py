import numpy as np

from src.errors import Error
from src.Optimizer import Optimizer
from src.Training import Training


class MultiLayerPerceptron:

    def __init__(
        self,
        training_method: Training,
        optimizer: Optimizer,
        neural_network: Training.NeuralNetwork,
        error: Error,
        epochs: int,
        learning_rate: float,
    ):
        self.training_method = training_method
        self.epochs = epochs
        self.neural_network = neural_network
        self.error = error
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    @staticmethod
    def predict(network: Training.NeuralNetwork, input_matrix: np.array):
        output = input_matrix
        for layer in network:
            output = layer.forward(output)

        return output

    def train(self, input_matrix, expected_output):
        self.training_method.train(
            self.neural_network,
            self.error,
            input_matrix,
            expected_output,
            self.epochs,
            self.learning_rate,
        )
