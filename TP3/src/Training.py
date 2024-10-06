from abc import ABC, abstractmethod
import numpy as np


class Training(ABC):
    @abstractmethod
    def train(self, network, error, X, Y, epochs, learning_rate):
        pass

#Actualizacion pesos--> luego de calcular el dW para TODOS LOS ELEMENTOS del conjunto de datos
class Batch(Training):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def train(self, network, error, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                Y_batch = Y[i:i + self.batch_size]

                # forward
                output = X_batch
                for layer in network:
                    output = layer.forward(output)

                # error
                loss = np.mean(error.error( Y_batch, output))

                # backward
                output_gradient = error.error_prime( Y_batch, output)
                for layer in reversed(network):
                    output_gradient = layer.backward(output_gradient, learning_rate)

                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network
    
#Actualizacion pesos--> luego de calcular el dW para UN SUBCONJUNTO DE ELEMENTOS del conj de datos
class MiniBatch(Training):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def train(self, network, error, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                Y_batch = Y[i:i + self.batch_size]

                # forward
                output = X_batch
                for layer in network:
                    output = layer.forward(output)

                # error
                loss = np.mean(error.error( Y_batch, output))

                # backward
                output_gradient = error.error_prime(Y_batch, output)
                for layer in reversed(network):
                    output_gradient = layer.backward(output_gradient, learning_rate)

                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network
    
#actualizacion pesos de la red--> luego de calcular el dW PARA UN ELEMENTO del conjunto de datos
class Online(Training):
    def train(self, network, error, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                X_batch = X[i]
                Y_batch = Y[i]

                # forward
                output = X_batch
                for layer in network:
                    output = layer.forward(output)

                # error
                loss = np.mean(error.error( Y_batch, output))

                # backward
                output_gradient = error.error_prime( Y_batch, output)
                for layer in reversed(network):
                    output_gradient = layer.backward(output_gradient, learning_rate)

                print(f"Epoch {epoch + 1}/{epochs} - loss: {loss}")

        return network