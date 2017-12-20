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
class PlainModel(object):
    def __init__(self):
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


class GraphModel(object):
    def __init__(self):
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

    def fit(self, X, Y):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def update_parameters(self):
        pass
