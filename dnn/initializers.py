import numpy as np


def with_zeros(shape, *args):
    """
    Initializes with zeros, expects shape tuple.
    """
    return np.zeros(shape)


def with_random(shape, *args):
    """
    Initializes with small random values, expects shape tuple.
    """
    return np.random.randn(*shape) * 0.01


def with_he(shape, prev_units):
    """
    He Initialization, expects shape tuple and prev_units.
    Recommended for ReLU.
    """
    return np.random.randn(*shape) * np.sqrt(2. / prev_units)


def with_xavier(shape, prev_units):
    """
    Xavier Initialization with small random values, expects shape tuple and prev_units.
    """
    return np.random.randn(*shape) * np.sqrt(1. / prev_units)
