import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from src.errors import Error
from src.Layer import Layer


class Training(ABC):
    type NeuralNetwork = list[Layer]

    def __init__(self, predict: Callable[[NeuralNetwork, np.array], np.array]):
        self.predict = predict

    @abstractmethod
    def train(
        self,
        network: NeuralNetwork,
        error: Error,
        input_matrix: np.array,
        expected_output_matrix: np.array,
        epochs: int = 10_000,
        learning_rate: float = 0.1,
    ):
        pass


# Actualizacion pesos--> luego de calcular el dW para TODOS LOS ELEMENTOS del conjunto de datos
class Batch(Training):
    def __init__(
        self,
        predict: Callable[[Training.NeuralNetwork, np.array], np.array],
        batch_size: int,
    ):
        super().__init__(predict)
        self.batch_size = batch_size

    def train(
        self,
        network: Training.NeuralNetwork,
        error: Error,
        input_matrix: np.array,
        expected_output_matrix: np.array,
        epochs: int = 10_000,
        learning_rate: float = 0.1,
    ):
        for epoch in range(epochs):
            for i in range(0, input_matrix.shape[0], self.batch_size):
                X_batch = input_matrix[i : i + self.batch_size]
                Y_batch = expected_output_matrix[i : i + self.batch_size]

                # forward
                output = self.predict(network, X_batch)

                # error
                loss = np.mean(error.error(Y_batch, output))

                # backward
                output_gradient = error.error_prime(Y_batch, output)
                for layer in reversed(network):
                    output_gradient = layer.backward(output_gradient, learning_rate)

                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network


# Actualizacion pesos--> luego de calcular el dW para UN SUBCONJUNTO DE ELEMENTOS del conj de datos
class MiniBatch(Training):
    def __init__(
        self,
        predict: Callable[[Training.NeuralNetwork, np.array], np.array],
        batch_size: int,
    ):
        super().__init__(predict)
        self.batch_size = batch_size

    def train(
        self,
        network: Training.NeuralNetwork,
        error: Error,
        input_matrix: np.array,
        expected_output_matrix: np.array,
        epochs: int = 10_000,
        learning_rate: float = 0.1,
    ):
        for epoch in range(epochs):
            for i in range(0, input_matrix.shape[0], self.batch_size):
                X_batch = input_matrix[i : i + self.batch_size]
                Y_batch = expected_output_matrix[i : i + self.batch_size]

                # forward
                output = self.predict(network, X_batch)

                # error
                loss = np.mean(error.error(Y_batch, output))

                # backward
                output_gradient = error.error_prime(Y_batch, output)
                for layer in reversed(network):
                    output_gradient = layer.backward(output_gradient, learning_rate)

                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network


# actualizacion pesos de la red--> luego de calcular el dW PARA UN ELEMENTO del conjunto de datos
class Online(Training):
    def __init__(self, predict: Callable[[Training.NeuralNetwork, np.array], np.array]):
        super().__init__(predict)

    def train(
        self,
        network: Training.NeuralNetwork,
        error: Error,
        input_matrix: np.array,
        expected_output_matrix: np.array,
        epochs: int = 10_000,
        learning_rate: float = 0.1,
    ):
        for epoch in range(epochs):
            for x, _ in zip(input_matrix, expected_output_matrix):
                # forward
                output = self.predict(network, x)

                # error
                loss = error.error(expected_output_matrix, output)

                # backward
                output_gradient = error.error_prime(expected_output_matrix, output)
                for layer in reversed(network):
                    output_gradient = layer.backward(output_gradient, learning_rate)

                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network
