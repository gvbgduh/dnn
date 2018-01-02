import numpy as np

from .initializers import with_he


"""
TODOs:
 * Pretify W, b initing
 * W, b -> during fitting

 * Better exception handling
"""

class LayersChainIsBroken(Exception):
    pass


class ParametersAreNotInitialized(Exception):
    pass


class NoDerivativeProvided(Exception):
    pass


class Layer(object):
    """
    Generic base class for a layer in the feedforward neural network.
    Params:
     * units (int) - number of units in the layer
     * activation (function object) - activation function
    """
    def __init__(self, units, activation, input_data=[], initializer=with_he, idx=None, input_dim=None, learning_rate=None):
        self.idx = idx                  # Index of the layer, assigned in the model
        self.input_dim = input_dim
        self.units = units              # Number of units
        self.activation = activation    # Activation function for units
        self.input_data = input_data    # Input vector X of A[l]
        self.initializer = initializer  # Init func

        self.learning_rate = learning_rate

        self.next = None                # Next layer in the chain, assigned in the model
        self.prev = None                # Previous layer in the chain, assigned in the model

        self.W = None
        self.b = None

        self.Z = None
        self.A = None

        self.dW = None
        self.db = None

        self.dA = None
        self.dZ = None                  # Set externally

    @property
    def grads(self):
        return {
            'dZ': self.dZ,
            'dA': self.dA,
            'dW': self.dW,
            'db': self.db
        }

    def __repr__(self):
        return 'Layer {}: {} units [{}]'.format(self.idx, self.units, self.activation.name)

    def initialize(self):
        if not self.prev and not self.input_dim:
            raise LayersChainIsBroken('No prev layer and an input size')
        if self.prev and self.input_dim:
            raise LayersChainIsBroken('Only the first layer should have an input size')

        if self.input_dim:
            self.W = self.initializer((self.units, self.input_dim), self.input_dim)
            self.b = np.random.rand(self.units, 1) * 0.01
        else:
            self.W = self.initializer((self.units, self.prev.units), self.prev.units)
            self.b = np.random.rand(self.units, 1) * 0.01

    def forward(self):
        if not self.prev and not self.input_dim:
            raise LayersChainIsBroken('No prev layer and an input size')
        if self.prev and self.input_dim:
            raise LayersChainIsBroken('Only the first layer should have an input size')
        if self.W is None or self.b is None:
            raise ParametersAreNotInitialized

        # Check input data provided

        if self.input_dim:
            self.Z = np.dot(self.W, self.input_data) + self.b
        else:
            self.Z = np.dot(self.W, self.prev.A)

        self.A = self.activation.forward(self.Z)

    def backward(self):
        # import pdb ; pdb.set_trace()
        if self.dA is None:
            raise NoDerivativeProvided('No dA provided')
        if self.prev is None and self.idx != 1:
            raise LayersChainIsBroken('Chain is broken or attempt to backprop the input layer')
        if self.prev and self.prev.A is None and self.idx != 1:
            raise LayersChainIsBroken('Chain is broken or no activation in the previous layer')
        # import pdb ; pdb.set_trace()
        m = self.prev.A.shape[1] if self.idx != 1 else self.input_dim

        self.dZ = self.activation.backward(self.dA, self.Z)


        self.db = (1. / m) * np.sum(self.dZ, axis=1, keepdims=True)
        if self.idx != 1:
            self.prev.dA = np.dot(self.W.T, self.dZ)
            self.dW = (1. / m) * np.dot(self.dZ, self.prev.A.T)
            assert (self.prev.dA.shape == self.prev.A.shape)
        else:
            self.dW = (1. / m) * np.dot(self.dZ, self.input_data.T)

        assert (self.dW.shape == self.W.shape)
        assert (self.db.shape == self.b.shape)

    def update_params(self, learning_rate):
        if self.learning_rate is not None:
            learning_rate = self.learning_rate

        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db
