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
        self.epsilon = 0.001

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
            output_gradient = 0.0
            for x, y in zip(input_matrix, expected_output_matrix):
                # forward
                output = self.predict(network, x)

                # error
                loss = error.error(y, output)
                total_loss += loss

                # backward
                output_gradient = error.error_prime(y, output)

                for layer in reversed(network):
                    # print(f'Weights Gradient: {weights_gradient}')
                    output_gradient = layer.backward(
                        output_gradient, learning_rate
                    )
            for layer in reversed(network):
                layer.update()
            total_error = total_loss/len(input_matrix)
            errors.append(total_error)
            if total_error < self.epsilon:
                break

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
        self.epislon = 0.001

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

                for x, y in zip(X_batch, Y_batch):
                    # forward
                    output = self.predict(network, x)

                    # error
                    loss = error.error(y, output)
                    total_loss += loss

                    # backward
                    output_gradient = error.error_prime(y, output)
                    for layer in reversed(network):
                        output_gradient = layer.backward(
                            output_gradient, learning_rate
                        )
                for layer in reversed(network):
                    layer.update()

            total_error = total_loss/len(input_matrix)
            errors.append(total_error)
            if total_error < self.epsilon:
                break

        print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")
        return network, errors


# actualizacion pesos de la red--> luego de calcular el dW PARA UN ELEMENTO del conjunto de datos
class Online(Training):
    def __init__(self, predict: Callable[[Training.NeuralNetwork, np.array], np.array]):
        super().__init__(predict)
        self.epsilon = 0.001

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
                for layer in reversed(network):
                    output_gradient = layer.backward(
                        output_gradient, learning_rate
                    )
                for layer in reversed(network):
                    layer.update()

            total_error = total_loss/len(input_matrix)
            errors.append(total_error)
            if total_error < self.epsilon:
                break

        print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")
        return network, errors
