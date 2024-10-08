#   ------          ----
#   | xi |----------| yj | ---> (todas las xi con todas las xj): y1 = x1.w11 + x2.w12+...+xi.w1i + b1
#   ------    wji   -----                                          (b1= bias)
#
# Todas esas ecuaciones pasa a matriz (Y = W . X + B) (matrices)
from pickletools import optimize

from .Layer import Layer
import numpy as np
from .Optimizer import Optimizer, GradientDescent, Momentum


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        optimizer: Optimizer,
    ):
        """

        Parameters
        ----------
        input_size: Number of input neurons
        output_size: Number of output neurons
        optimizer: An implementation of Optimizer
        """
        super().__init__()

        # W Matrix
        self.weights = np.random.randn(output_size, input_size)
        # B Matrix
        self.bias = np.random.randn(output_size, 1)

        self.optimizer = optimizer
        self.weights_accum = 0
        self.bias_accum = 0
        self.count = 0

    def forward(self, input_matrix):
        """
        Performs Y = W. X + B

        Parameters
        ----------
        input_matrix

        Returns
        -------
        Y
        """
        self.input_matrix = input_matrix
        return np.dot(self.weights, self.input_matrix) + self.bias

    # TODO sacar el learning_Rate(me reta layer), ver si se puede sacar de layer o no
    def backward(self, output_gradient, learning_rate):
        """
        Returns dE/dX

        Parameters
        ----------
        output_gradient: dE/dY
        learning_rate:

        Returns
        -------

        """
        weights_gradient = np.dot(output_gradient, self.input_matrix.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # dE/dW = dE/dY . dY/dW = dE/dY . X^t
        weights, bias = self.optimizer.update(
            self.weights, self.bias, weights_gradient, output_gradient
        )
        self.weights_accum += weights
        self.bias_accum += bias
        self.count += 1

        # dE/dX = dE/dY . dY/dX = W^t . dE/dY
        return input_gradient

    def update(self):
        self.weights -= self.weights_accum / self.count
        self.bias -= self.bias_accum / self.count
        self.count = 0
        self.weights_accum = 0
        self.bias_accum = 0
        if isinstance(self.optimizer, Momentum):
            self.optimizer.deltaWeights = None
            self.optimizer.deltaBias = None
