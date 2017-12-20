import numpy as np
import unittest as ut

from activations import relu, identity
from layers import Layer
from models import GraphModel, PlainModel


class TestGraphModel(ut.TestCase):
    """
    Let's test the net on parabola
    """
    def setUp(self):
        np.random.seed(1)
        self.X = np.arange(-100, 100)
        self.Y = self.X ** 2
        self.model = GraphModel()

    def test_model_creation(self):
        self.assertEqual(self.model.show(), 'No layers')
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.assertEqual(self.model.depth, 3)
        self.assertEqual(
            self.model.show(),
            'X -> Layer 1: 20 units [relu] -> Layer 2: 10 units [relu] -> Layer 3: 1 units [identity] -> y'
        )

    def test_model_init(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()

        # Check dims
        self.assertEqual(self.model.first.W.shape, (20, self.X.shape[0]))
        self.assertEqual(self.model.first.next.W.shape, (10, 20))
        self.assertEqual(self.model.first.next.next.W.shape, (1, 10))
        self.assertEqual(self.model.first.b.shape, (20, 1))
        self.assertEqual(self.model.first.next.b.shape, (10, 1))
        self.assertEqual(self.model.first.next.next.b.shape, (1, 1))
        self.assertEqual(self.model.first.next.next.next, None)

    def test_forward_prop(self):
        pass

    def test_backward_prop(self):
        pass

    def test_optimization(self):
        pass

    def test_gradient_checking(self):
        pass


class TestPlainModel(ut.TestCase):
    """
    Let's test the net on parabola
    """
    def setUp(self):
        np.random.seed(1)
        self.X = np.arange(-100, 100)
        self.Y = self.X ** 2
        self.model = PlainModel()

    def test_model_creation(self):
        self.assertEqual(self.model.show(), 'No layers')
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.assertEqual(self.model.depth, 3)
        self.assertEqual(
            self.model.show(),
            'X -> Layer 1: 20 units [relu] -> Layer 2: 10 units [relu] -> Layer 3: 1 units [identity] -> y'
        )

    def test_model_init(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()

        # Check dims
        self.assertEqual(self.model.layers[0].W.shape, (20, self.X.shape[0]))
        self.assertEqual(self.model.layers[1].W.shape, (10, 20))
        self.assertEqual(self.model.layers[2].W.shape, (1, 10))
        self.assertEqual(self.model.layers[0].b.shape, (20, 1))
        self.assertEqual(self.model.layers[1].b.shape, (10, 1))
        self.assertEqual(self.model.layers[2].b.shape, (1, 1))

    def test_forward_prop(self):
        pass

    def test_backward_prop(self):
        pass

    def test_optimization(self):
        pass

    def test_gradient_checking(self):
        pass
