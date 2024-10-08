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
        return (
            self.learning_rate * weights_gradient,
            self.learning_rate * bias_gradient,
        )


class Momentum(Optimizer):

    def __init__(self, learning_rate, alpha=0.8):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.deltaWeights = None
        self.deltaBias = None

    def update(self, weights, bias, weights_gradient, bias_gradient):
        # First time init
        if self.deltaWeights is None:
            self.deltaWeights = np.zeros(weights.shape)
        if self.deltaBias is None:
            self.deltaBias = np.zeros(bias.shape)

        self.deltaWeights = (
            self.alpha * self.deltaWeights - self.learning_rate * weights_gradient
        )
        self.deltaBias = (
            self.alpha * self.deltaBias - self.learning_rate * bias_gradient
        )

        return (self.deltaWeights, self.deltaBias)


class Adam(Optimizer):

    # good default settings:
    def __init__(
        self, learning_rate: float, beta1: float, beta2: float, epsilon: float
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_weights = None
        self.v_weights = None
        self.m_bias = None
        self.v_bias = None
        self.t = 0

    def update(self, weights, bias, weights_gradient, bias_gradient):
        # First time init
        if self.m_weights is None:
            self.m_weights = np.zeros(weights.shape)
        if self.v_weights is None:
            self.v_weights = np.zeros(weights.shape)
        if self.m_bias is None:
            self.m_bias = np.zeros(bias.shape)
        if self.v_bias is None:
            self.v_bias = np.zeros(bias.shape)

        self.t += 1

        # Gradients: parameters

        # Biased first moment estimation
        self.m_weights = (
            self.beta1 * self.m_weights + (1 - self.beta1) * weights_gradient
        )
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * bias_gradient

        # Biased second moment estimation
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(
            weights_gradient
        )
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * np.square(
            bias_gradient
        )

        # First moment
        m_weights_hat = self.m_weights / (1 - self.beta1**self.t)
        m_bias_hat = self.m_bias / (1 - self.beta1**self.t)
        # Second moment
        v_weights_hat = self.v_weights / (1 - self.beta2**self.t)
        v_bias_hat = self.v_bias / (1 - self.beta2**self.t)

        # Update weights and bias
        updated_weights_deltas = (
            self.learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
        )
        updated_biases_deltas = self.learning_rate * m_bias_hat / (np.sqrt(v_bias_hat) + self.epsilon)

        return (updated_weights_deltas, updated_biases_deltas)
