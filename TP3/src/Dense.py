#   ------          ----
#   | xi |----------| yj | ---> (todas las xi con todas las xj): y1 = x1.w11 + x2.w12+...+xi.w1i + b1
#   ------    wji   -----                                          (b1= bias)
#
# Todas esas ecuaciones pasa a matriz (Y = W . X + B) (matrices)

from .Layer import Layer
import numpy as np
from .Optimizer import Optimizer, GradientDescent


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        optimizer: Optimizer = GradientDescent(0.01),
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
        self.input = input_matrix
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate: float):
        """
        Returns dE/dX

        Parameters
        ----------
        output_gradient: dE/dY
        learning_rate:

        Returns
        -------

        """

        # dE/dW = dE/dY . dY/dW = dE/dY . X^t
        weights_gradient = np.dot(output_gradient, self.input.T)
        # dE/dX = dE/dY . dY/dX = W^t . dE/dY
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights, self.bias = self.optimizer.update(
            self.weights, self.bias, weights_gradient, output_gradient
        )
        return input_gradient
