"""
Main flow:

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

TODO's:
* Generalize common bits to lower level abstract model.
"""
import numpy as np


class LayersNotProvided(Exception):
    pass


class PlainModel(object):
    def __init__(self, cost, learning_rate):
        self.cost = cost
        self.learning_rate = learning_rate

        self.costs = []
        self.layers = []

    def __repr__(self):
        return 'Model: {} layers'.format(self.depth)

    @property
    def depth(self):
        return len(self.layers)

    def show(self):
        if not self.layers:
            return 'No layers'

        arch = 'X -> {} -> y'.format(' -> '.join(map(str, self.layers)))
        return arch

    def add_layer(self, layer):
        if self.layers:
            layer.prev = self.layers[-1]
        self.layers.append(layer)
        layer.idx = self.depth

    def initialize(self):
        # TODO Use multithreading pool!?
        for l in self.layers:
            l.initialize()

    def forward_prop(self, X):
        if not self.layers:
            raise LayersNotProvided

        self.layers[0].input_data = X
        for layer in self.layers:
            layer.forward()

    def get_cost(self, Y):
        cur_cost = self.cost.forward(self.layers[-1].A, Y)
        self.costs.append(cur_cost)
        return cur_cost

    def backward_prop(self, Y):

        assert (self.layers[-1].A.shape == Y.shape)
        # Init so?
        # dAL = - (np.divide(Y, self.layers[-1].A) - np.divide(1 - Y, 1 - self.layers[-1].A))
        self.layers[-1].dA = self.cost.backward(self.layers[-1].A, Y)
        # self.layers[-1].dA = dAL
        for layer in self.layers[::-1]:
            # import pdb ; pdb.set_trace()
            layer.backward()

    def update_parameters(self):
        for layer in self.layers:
            layer.update_params(self.learning_rate)

    def fit(self, X, Y, epoches):
        # Checks
        
        self.initialize()
        for i in range(epoches):
            # import pdb; pdb.set_trace()
            if i != 0:
                self.update_parameters()
            self.forward_prop(X)
            cur_cost = self.get_cost(Y)
            self.backward_prop(Y)
            print('|' + '-' * 27 + '| ' + 'Cost: {:.4f}'.format(cur_cost))


class GraphModel(object):
    def __init__(self, cost):
        self.cost = cost
        self.first = None
        self.last = None
        self.depth = 0

    def __repr__(self):
        return 'Model: {} layers'.format(self.depth)

    def show(self):
        if self.first is None:
            return 'No layers'

        cur = self.first
        arch = 'X -> {} -> '.format(cur)
        while cur.next:
            cur = cur.next
            arch += '{} -> '.format(cur)
        return arch + 'y'

    def add_layer(self, layer):
        self.depth += 1
        layer.idx = self.depth

        if self.first is None:
            self.first = layer
            self.last = layer
        else:
            self.last.next = layer
            layer.prev = self.last
            self.last = layer

    def initialize(self):
        cur = self.first
        while cur:
            cur.initialize()
            cur = cur.next

    def forward_prop(self, X):
        if self.first is None:
            raise LayersNotProvided

        self.first.input_data = X
        cur = self.first
        while cur:
            cur.forward()
            cur = cur.next

    def backward_prop(self):
        pass

    def update_parameters(self):
        pass

    def fit(self, X, Y):
        pass
