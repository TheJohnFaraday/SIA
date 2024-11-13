import numpy as np
from abc import ABC, abstractmethod


class Error(ABC):
    @abstractmethod
    def error(self, y_true, y_pred):
        pass

    @abstractmethod
    def error_prime(self, y_true, y_pred):
        pass


class MSE(Error):
    def error(self, y_true, y_pred):
        return np.mean(np.power((y_true - y_pred), 2))

    def error_prime(self, y_true, y_pred):
        return (
            2
            * (y_pred - y_true)
            / np.size(y_true)
        )
