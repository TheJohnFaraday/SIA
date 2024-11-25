import numpy as np

from typing import List
from src.errors import Error
from src.Optimizer import Adam
from src.Dense import Dense
from src.activation_functions import Tanh
from src.MultiLayerPerceptron import MultiLayerPerceptron
from src.Training import Batch


class Autoencoder:
    def __init__(
        self,
        input_size: int,
        list_size: int,
        layers: List[int],
        latent_space_dim,
        error: Error,
        epochs: int,
        beta: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        learning_rate: float,
        is_variational: bool = False,
    ):
        self.error = error
        self.learning_rate = learning_rate
        encoder_layers = []
        prior_layer = input_size
        for layer in layers:
            print(f"prior layer {prior_layer} -> layer {layer}")
            encoder_layers.append(
                Dense(
                    input_size=prior_layer,
                    output_size=layer,
                    optimizer=Adam(
                        learning_rate=learning_rate,
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=epsilon,
                    ),
                )
            )
            encoder_layers.append(Tanh(beta=beta))
            prior_layer = layer
        latent_dim = Dense(
                input_size=prior_layer,
                output_size=latent_space_dim,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            )
        encoder_layers.append(latent_dim)
        encoder_layers.append(Tanh(beta=beta))

        self.latent_dim = latent_dim if is_variational else None
        self.encoder_layers = encoder_layers

        prior_layer = latent_space_dim
        decoder_layers = []

        for layer in layers[::-1]:
            print(f"prior layer {prior_layer} -> layer {layer}")
            decoder_layers.append(
                Dense(
                    input_size=prior_layer,
                    output_size=layer,
                    optimizer=Adam(
                        learning_rate=learning_rate,
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=epsilon,
                    ),
                )
            )
            decoder_layers.append(Tanh(beta=beta))
            prior_layer = layer
        decoder_layers.append(
            Dense(
                input_size=prior_layer,
                output_size=input_size,
                optimizer=Adam(
                    learning_rate=learning_rate,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                ),
            )
        )
        decoder_layers.append(Tanh(beta=beta))

        self.decoder_layers = decoder_layers
        self.neural_network = self.encoder_layers + self.decoder_layers
        self.autoencoder = MultiLayerPerceptron(
            training_method=Batch(
                MultiLayerPerceptron.predict,
                batch_size=list_size,  # Batch size igual al número de caracteres del subset
                epsilon=epsilon,
                is_variational=is_variational,
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
            self.latent_dim
        )
