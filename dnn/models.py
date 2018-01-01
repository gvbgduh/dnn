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
class LayersNotProvided(Exception):
    pass


class PlainModel(object):
    def __init__(self, cost):
        self.layers = []
        self.cost = cost
        self.costs = []

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
        for l in self.layers:
            l.forward()

    def get_cost(self, Y):
        cur_cost = self.cost.forward(self.layers[-1].A, Y)
        self.costs.append(cur_cost)
        return cur_cost
    
    def back_prop(self, Y):
        l = self.layers[-1]
        l.dZ = self.cost.backward(l.A, Y)

    def fit(self, X, Y):
        # Checks
        self.initialize()
        self.forward_prop(X)
        self.get_cost(Y)
        self.back_prop(Y)


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

    def backward(self):
        pass

    def update_parameters(self):
        pass

    def fit(self, X, Y):
        pass
