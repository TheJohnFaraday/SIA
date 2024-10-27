import logging

import numpy as np
import pandas as pd

from src.configuration import OjaConfig


class OjaSimplePerceptron:
    def __init__(
        self,
        configuration: OjaConfig,
        input_data: pd.DataFrame,
    ):
        data_size = len(list(input_data.to_numpy())[0])

        self.__configuration = configuration

        # initialize weights w to small random values
        self.__weights = np.random.uniform(
            low=configuration.weights_low,
            high=configuration.weights_high,
            size=data_size,
        )
        self.__learning_rate = configuration.learning_rate
        self.__delta_weights = np.zeros(data_size)
        self.__input_data = input_data
        self.__output = 0.0

    @staticmethod
    def activation_function(x: float):
        return x

    @staticmethod
    def activation_prime(x: float):
        return 1

    def predict(self, input_data: np.ndarray[np.float64]):
        weighted_sum = self.__weights @ input_data
        return self.activation_function(weighted_sum)

    def train(self):
        for current_epoch in range(self.__configuration.max_epochs):
            # for x, y in zip(input_data, expected_output):
            for data in self.__input_data.to_numpy(dtype=np.float64):
                # forward
                self.__output = self.predict(data)

                # Oja's learning rule
                self.__delta_weights += self.__learning_rate * (
                    self.__output * data - (self.__output**2) * self.__weights
                )

                self.__weights = self.__delta_weights + self.__weights
                self.__delta_weights.fill(0.0)

                logging.debug(f"Epoch: {current_epoch}")

        return self.__output

    @property
    def output(self):
        return self.__output

    @property
    def weights(self):
        return self.__weights
