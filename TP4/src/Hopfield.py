import numpy as np


class Hopfield:
    def __init__(self, N=25):
        self.N = N
        self.w = np.zeros((N, N))
        self.xi = np.empty((N, 0), dtype='int8')
        self.S = -1 * np.ones((N, 0), dtype='int8')
        self.p = 0  # Number of saved patterns
        self.t = []  # Steps

    def set_patterns(self, input_pattern):
        w = np.outer(input_pattern, input_pattern.T) / self.N
        np.fill_diagonal(w, 0)
        self.w += w
        self.xi = np.column_stack((self.xi, input_pattern))
        self.p = self.xi.shape[1]

    def set_initial_state(self, pattern):
        self.t = []
        self.S = pattern

    def train(self):
        while True:
            second_last_state = np.copy(self.S)
            for i in range(2):
                last_state = np.copy(self.S)
                self.S = self.sign_0(np.dot(self.w, self.S))
                if np.array_equal(last_state, self.S):
                    return (self.S, self.t)
                self.t.append(self.S)
            if np.array_equal(second_last_state, self.S):
                return (self.S, self.t)

    def sign_0(self, arr):
        return np.where(arr >= -1e-15, 1, -1)
