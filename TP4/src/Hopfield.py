import numpy as np


class Hopfield:
    def __init__(self, patterns):
        ps_len = len(patterns)
        self.patterns = np.reshape(patterns, (ps_len, 25, 1))
        self.weights = 1/ps_len * self.patterns * self.patterns.T
        for i in range(0, 25):
            self.weights[i][i] = 0
