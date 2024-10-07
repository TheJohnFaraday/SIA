from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, weights, bias, weights_gradient, bias_gradient):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, weights, bias, weights_gradient, bias_gradient):
        weights -= self.learning_rate * weights_gradient
        bias -= self.learning_rate * bias_gradient
        return weights, bias


class Momentum(Optimizer):

    #valores usuales de alpha 0.8/0.9 y es entre 0 y 1
    def __init__(self, learning_rate, alpha):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.deltaW = None
        self.deltaB = None

    def update(self, weights, bias, weights_gradient, bias_gradient):
        if self.deltaW is None:
            self.deltaW = np.zeros(weights.shape)
            self.deltaB = np.zeros(bias.shape)

        #TODO chequear formulas y el tema de acumulación de los gradients
        self.deltaW = - self.learning_rate * weights_gradient + self.alpha * self.deltaW
        self.deltaB = - self.learning_rate * bias_gradient + self.alpha * self.deltaB

        weights += self.deltaW
        bias += self.deltaB

        return weights, bias


class Adam(Optimizer):

    #good default settings:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.mw = None
        self.vw = None
        self.mb = None
        self.vb = None
        self.t = 0

    def update(self, weights, bias, weights_gradient, bias_gradient):

        self.t += 1

        self.mw = self.beta1 * self.mw + (1 - self.beta1) * weights_gradient
        self.vw = self.beta2 * self.vw + (1 - self.beta2) * weights_gradient**2

        self.mw_hat = self.mw / (1 - self.beta1**self.t)
        self.vw_hat = self.vw / (1 - self.beta2**self.t)

        self.mb = self.beta1 * self.mb + (1 - self.beta1) * bias_gradient
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * bias_gradient**2

        self.mb_hat = self.mb / (1 - self.beta1**self.t)
        self.vb_hat = self.vb / (1 - self.beta2**self.t)

        weights -= self.learning_rate * self.mw_hat / (np.sqrt(self.vw_hat) + 1e-8)
        bias -= self.learning_rate * self.mb_hat / (np.sqrt(self.vb_hat) + 1e-8)

        return weights, bias







