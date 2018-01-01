import numpy as np


class CostFunction(object):
    def __init__(self, name, forward, backward):
        self.name = name
        self.forward = forward
        self.backward = backward


def cross_entropy_cost(AL, Y):
    """
    Implements the cost cross-entropy function.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), (1 - Y))
    cost = - (1 / m) * np.sum(logprobs)
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


def _MSE_forward(AL, Y):
    m = len(Y)
    cost = (1 / (2 * m)) * np.sum((AL - Y)**2)
    return cost


def _MSE_backward(AL, Y):
    return AL - Y


MSE = CostFunction('MSE', _MSE_forward, _MSE_backward)
