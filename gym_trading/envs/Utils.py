import numpy as np


def range_normalization(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-20)


def standard_score_normalization(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    return (x - x_mean) / (x_std + 1e-20)
