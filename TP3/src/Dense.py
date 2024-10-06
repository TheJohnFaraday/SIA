#   ------          ----
#   | xi |----------| yj | ---> (todas las xi con todas las xj): y1 = x1.w11 + x2.w12+...+xi.w1i + b1
#   ------    wji   -----                                          (b1= bias)
#
# Todas  esas ecuaciones pasa a matriz (Y = W . X + B) (matrices)

from .Layer import Layer
import numpy as np
from .Optimizer import GradientDescent

class Dense(Layer):
    #input_size = numero de neuronas en el input (idem output_size)
    def __init__(self, input_size, output_size, optimizer=GradientDescent(0.01)):
        #matriz: W
        self.weights = np.random.randn(output_size, input_size)
        #matriz: B
        self.bias = np.random.randn(output_size, 1)
        
        self.optimizer = optimizer

    def forward(self, input):
        self.input = input
        #hace el calculo de (Y = W. X + B)
        return np.dot(self.weights, self.input) + self.bias

    #dE/dB = dE/dY . dY/dB = dE/dY . 1 = output_gradient
    def backward(self, output_gradient, learning_rate):
        #dE/dW = dE/dY . dY/dW = dE/dY . Xt (X transpuesta)
        weights_gradient = np.dot(output_gradient, self.input.T)
        #dE/dX = dE/dY . dY/dX = Wt . dE/dY
        input_gradient = np.dot(self.weights.T, output_gradient)
        #update de los parametros:
        self.weights, self.bias = self.optimizer.update(self.weights, self.bias, weights_gradient, output_gradient)
        return input_gradient


