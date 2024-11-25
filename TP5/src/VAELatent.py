import numpy as np

from src.Layer import Layer
from src.Dense import Dense
from src.Optimizer import Optimizer


class VAELatent(Layer):
    def __init__(self, input_size: int, output_size: int, optimizer: Optimizer):
        self.mean_p = Dense(input_size, output_size, optimizer)
        self.log_var_p = Dense(input_size, output_size, optimizer)
        self.output_size = output_size
        self.l = 2

    def forward(self, input):
        self.epsilon = np.random.standard_normal(size=(self.output_size, input.shape[1]))
        self.log_var = self.log_var_p.forward(input)
        self.mean = self.mean_p.forward(input)
        self.sample = np.exp(0.5 * self.log_var) * self.epsilon + self.mean

        return self.sample

    def backward(self, output_gradient, learning_rate):
        mean_gradient = self.mean_p.backward(output_gradient + self.l*self.mean, learning_rate)
        log_var_gradient = self.log_var_p.backward(output_gradient * self.epsilon + self.l*(np.exp(self.log_var) - 1), learning_rate)

        return mean_gradient + log_var_gradient

    def get_kl(self):
        return -0.5 * np.sum(1 + self.log_var - np.square(self.mean) - np.exp(self.log_var)) * self.l

    def update(self):
        self.log_var_p.update()
        self.mean_p.update()
