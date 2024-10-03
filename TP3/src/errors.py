import numpy as np


def mse(y_true, y_pred) -> (float, float):
    mse = np.mean(np.power(y_true - y_pred, 2))
    mse_prime = 2 * (y_pred - y_true) / np.size(y_true)
    return (mse, mse_prime)
