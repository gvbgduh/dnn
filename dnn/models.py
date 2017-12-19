"""
def initialize_parameters_deep(layer_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters

[12288, 20, 7, 5, 1]
"""

class Model(object):
    def __init__(self):
        self.first = None
        self.last = None
    
    def add_layer(self, layer):
        if self.first is None:
            self.first = layer
            self.last = layer
        else:
            self.last.next = layer
            layer.prev = self.last
            self.last = layer

    def fit(self, X, Y):
        pass
    
    def forward(self):
        pass

    def backward(self):
        pass
    
    def update_parameters(self):
        pass
