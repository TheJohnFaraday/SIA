from .Activation import Activation
import numpy as np


# tanh(x), tanh'(x)= 1 - tanh^2(x)
class Tanh(Activation):
    def __init__(self, beta: float = 0.4):
        def tanh(x, beta: float):
            return np.tanh(np.float128(x) * beta)

        def tanh_prime(x, beta: float):
            return 1 - tanh(x, beta) ** 2

        super().__init__(tanh, tanh_prime, beta)


# ReLU(x) = max(0, x), ReLU'(x) = 1 if x > 0 else 0
class ReLU(Activation):
    def __init__(self, beta: float = 1.0):
        def relu(x, beta: float):
            return np.maximum(0, np.float128(x))

        def relu_prime(x, beta: float):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime, beta)


class Logistic(Activation):
    def __init__(self, beta: float = 0.4):
        def logistic(x, beta: float):
            return 1 / (1 + np.exp(-2 * beta * np.float128(x)))

        def logistic_prime(x, beta: float):
            return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))

        super().__init__(logistic, logistic_prime, beta)
