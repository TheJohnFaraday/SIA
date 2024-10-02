from activation import Activation
import numpy as np

#tanh(x), tanh'(x)= 1 - tanh^2(x)
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)