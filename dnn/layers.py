from .initializers import with_he


"""
TODOs:
 * Pretify W, b initing
 * W, b -> during fitting
"""


class Layer(object):
    """
    Generic base class for a layer in the feedforward neural network.
    Params:
     * units (int) - number of units in the layer
     * activation (function object) - activation function 
    """
    def __init__(self, units, activation, input_data=[], initializer=with_he, prev=None):
        self.units = units              # Number of units
        self.activation = activation    # Activation function for units
        self.input = input_data         # Input vector X of A[l]
        self.initializer = initializer  # Init func  # TODO Add default

        self.next = None                # Next layer in the chain
        self.prev = prev                # Previous layer in the chain
        if prev is not None:            
            prev.next = self            # Link the prev layer to this
            # self.W = self.initializer((units, prev.units), prev.units)
            # self.b = self.initializer((units, prev.units), prev.units)
