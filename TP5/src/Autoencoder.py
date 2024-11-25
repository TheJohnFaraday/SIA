import numpy as np

from src.errors import Error
from src.Training import Training
from src.Optimizer import Adam
from src.Dense import Dense
from src.activation_functions import Tanh
from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Training import Batch


class Autoencoder:
    def __init__(
        self,
        input_size: int,
        error: Error,
        epochs: int,
        beta: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        learning_rate: float,
    ):
        subset_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ]  # Cambia los índices según los caracteres que quieras usar
        self.error = error
        self.learning_rate = learning_rate
        self.encoder_layers = [
            Dense(
                input_size=input_size,
                output_size=20,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            ),
            Tanh(beta=beta),
            Dense(
                input_size=20,
                output_size=15,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            ),
            Tanh(beta=beta),
            Dense(
                input_size=15,
                output_size=2,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            ),  # Espacio latente de dimensión 2
            Tanh(beta=beta),
        ]

        self.decoder_layers = [
            Dense(
                input_size=2,
                output_size=15,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            ),
            Tanh(beta=beta),
            Dense(
                input_size=15,
                output_size=20,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            ),
            Tanh(beta=beta),
            Dense(
                input_size=20,
                output_size=input_size,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            ),
            Tanh(beta=beta),
        ]
        self.neural_network = self.encoder_layers + self.decoder_layers
        self.autoencoder = MultiLayerPerceptron(
            training_method=Batch(
                MultiLayerPerceptron.predict,
                batch_size=len(
                    subset_indices
                ),  # Batch size igual al número de caracteres del subset
                epsilon=epsilon,
            ),
            neural_network=self.encoder_layers + self.decoder_layers,
            error=error,
            epochs=epochs,
            learning_rate=learning_rate,
        )

    def predict(self, input_matrix: np.array):
        return MultiLayerPerceptron.predict(
            self.autoencoder.neural_network, input_matrix
        )

    def encode(self, input_matrix: np.array):
        return MultiLayerPerceptron.predict(self.encoder_layers, input_matrix)

    def decode(self, input_matrix: np.array):
        return MultiLayerPerceptron.predict(self.decoder_layers, input_matrix)

    def train(self, input_matrix, expected_output):
        return self.autoencoder.train(
            input_matrix,
            expected_output,
        )
