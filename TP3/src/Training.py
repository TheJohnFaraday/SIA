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
    ) -> NeuralNetwork:
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
        errors = []
        for epoch in range(epochs):
            total_loss = 0
            weights_gradient = None
            output_gradient = 0.0
            for x, y in zip(input_matrix, expected_output_matrix):
                # forward
                output = self.predict(network, x)

                # error
                loss = error.error(y, output)
                total_loss += loss

                # backward
                output_gradient = error.error_prime(y, output)
                if weights_gradient is None:
                    weights_gradient = np.dot(output_gradient, x.T)
                else:
                    weights_gradient += np.dot(output_gradient, x.T)

            for layer in reversed(network):
                output_gradient, weights_gradient = layer.backward(
                    output_gradient, weights_gradient, learning_rate
                )

            errors.append(total_loss / len(input_matrix))
            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network, errors


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
        errors = []
        for epoch in range(epochs):
            total_loss = 0
            loss = 0
            output_gradient = 0.0
            for i in range(0, input_matrix.shape[0], self.batch_size):

                X_batch = input_matrix[i : i + self.batch_size]
                Y_batch = expected_output_matrix[i : i + self.batch_size]

                weights_gradient = None
                for x, y in zip(X_batch, Y_batch):
                    # forward
                    output = self.predict(network, x)

                    # error
                    loss = error.error(y, output)
                    total_loss += loss

                    # backward
                    output_gradient = error.error_prime(y, output)
                    if weights_gradient is None:
                        weights_gradient = np.dot(output_gradient, x.T)
                    else:
                        weights_gradient += np.dot(output_gradient, x.T)
                for layer in reversed(network):
                    output_gradient, weights_gradient = layer.backward(
                        output_gradient, weights_gradient, learning_rate
                    )

            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

            errors.append(total_loss / len(input_matrix))

        return network, errors


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
        errors = []
        for epoch in range(epochs):
            total_loss = 0
            loss = 0
            for x, y in zip(input_matrix, expected_output_matrix):
                # forward
                output = self.predict(network, x)

                # error
                loss = error.error(y, output)
                total_loss += loss

                # backward
                output_gradient = error.error_prime(y, output)
                weights_gradient = np.dot(output_gradient, x.T)
                for layer in reversed(network):
                    output_gradient, weights_gradient = layer.backward(
                        output_gradient, weights_gradient, learning_rate
                    )

            print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

            errors.append(total_loss / len(input_matrix))

        return network, errors
