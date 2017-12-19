import numpy as np
import unittest as ut

from ..activations import relu, identity
from ..layers import Layer
from ..models import Model


class TestModels(ut.TestCase):
    """
    Let's test the net on parabola
    """
    def setUp(self):
        np.random.seed(1)
        self.X = np.arange(-100, 100)
        self.Y = self.X ** 2
    
    def test_model_creation(self):
        import pdb; pdb.set_trace()
        model = Model()
        layer_1 = Layer(units=20, activation=relu)
        layer_2 = Layer(units=10, activation=relu)
        layer_3 = Layer(units=1, activation=identity)
        model.add_layer(layer_1)
        model.add_layer(layer_2)
        model.add_layer(layer_3)
