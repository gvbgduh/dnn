import numpy as np
import unittest as ut

from ..activations import relu, identity
from ..costs import MSE
from ..layers import Layer
from ..models import GraphModel, PlainModel


"""
TODO's
 * Check values as well
"""


class TestGraphModel(ut.TestCase):
    """
    Let's test the net on parabola
    """
    def setUp(self):
        np.random.seed(1)
        self.X = np.arange(-100, 100).reshape(1, -1)
        self.Y = self.X ** 2
        self.model = GraphModel(cost=MSE)

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
        self.assertEqual(self.model.first.b.shape, (20, 1))

        self.assertEqual(self.model.first.next.W.shape, (10, 20))
        self.assertEqual(self.model.first.next.b.shape, (10, 1))

        self.assertEqual(self.model.first.next.next.W.shape, (1, 10))
        self.assertEqual(self.model.first.next.next.b.shape, (1, 1))

        self.assertEqual(self.model.first.next.next.next, None)

    def test_forward_prop(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()
        self.model.forward_prop(self.X)

        # Chech dims
        self.assertEqual(self.model.first.Z.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.first.A.shape, (20, self.X.shape[1]))

        self.assertEqual(self.model.first.next.Z.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.first.next.A.shape, (10, self.X.shape[1]))

        self.assertEqual(self.model.first.next.next.Z.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.first.next.next.A.shape, (1, self.X.shape[1]))

        self.assertEqual(self.model.first.next.next.next, None)

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
        self.X = np.arange(-100, 100).reshape(1, -1)
        self.Y = self.X  # ** 2
        self.model = PlainModel(cost=MSE, learning_rate=0.000001)

    def test_model_creation(self):
        self.assertEqual(self.model.show(), 'No layers')
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[1]))
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
        self.assertEqual(self.model.layers[0].b.shape, (20, 1))

        self.assertEqual(self.model.layers[1].W.shape, (10, 20))
        self.assertEqual(self.model.layers[1].b.shape, (10, 1))

        self.assertEqual(self.model.layers[2].W.shape, (1, 10))
        self.assertEqual(self.model.layers[2].b.shape, (1, 1))

    def test_forward_prop(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()
        self.model.forward_prop(self.X)

        # Check dims
        self.assertEqual(self.model.layers[0].Z.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].A.shape, (20, self.X.shape[1]))

        self.assertEqual(self.model.layers[1].Z.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].A.shape, (10, self.X.shape[1]))

        self.assertEqual(self.model.layers[2].Z.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].A.shape, (1, self.X.shape[1]))

    def test_cost(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()
        self.model.forward_prop(self.X)

        self.assertEqual(
            [x.A.shape for x in self.model.layers],
            [(20, 200), (10, 200), (1, 200)]
        )

        cur_cost = self.model.get_cost(self.Y)
        self.assertEqual(cur_cost, 2029883751.7398474)
        self.assertEqual(len(self.model.costs), 1)

    def test_backward_prop(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()
        self.model.forward_prop(self.X)
        self.model.backward_prop(self.Y)

        # Check dims
        self.assertEqual(self.model.layers[2].W.shape, (1, 10))
        self.assertEqual(self.model.layers[2].b.shape, (1, 1))
        self.assertEqual(self.model.layers[2].dW.shape, (1, 10))
        self.assertEqual(self.model.layers[2].db.shape, (1, 1))

        self.assertEqual(self.model.layers[2].Z.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].A.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].dZ.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].dA.shape, (1, self.X.shape[1]))

        self.assertEqual(self.model.layers[1].W.shape, (10, 20))
        self.assertEqual(self.model.layers[1].b.shape, (10, 1))
        self.assertEqual(self.model.layers[1].dW.shape, (10, 20))
        self.assertEqual(self.model.layers[1].db.shape, (10, 1))

        self.assertEqual(self.model.layers[1].Z.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].A.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].dZ.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].dA.shape, (10, self.X.shape[1]))

        self.assertEqual(self.model.layers[0].W.shape, (20, self.X.shape[0]))
        self.assertEqual(self.model.layers[0].b.shape, (20, 1))
        self.assertEqual(self.model.layers[0].dW.shape, (20, self.X.shape[0]))
        self.assertEqual(self.model.layers[0].db.shape, (20, 1))

        self.assertEqual(self.model.layers[0].Z.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].A.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].dZ.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].dA.shape, (20, self.X.shape[1]))

    def test_backward_prop_with_update(self):
        self.model.add_layer(Layer(units=20, activation=relu, input_dim=self.X.shape[0]))
        self.model.add_layer(Layer(units=10, activation=relu))
        self.model.add_layer(Layer(units=1, activation=identity))
        self.model.initialize()
        self.model.forward_prop(self.X)
        cost_0 = self.model.get_cost(self.Y)
        self.model.backward_prop(self.Y)
        self.model.update_parameters()

        # Check dims
        self.assertEqual(self.model.layers[2].W.shape, (1, 10))
        self.assertEqual(self.model.layers[2].b.shape, (1, 1))
        self.assertEqual(self.model.layers[2].dW.shape, (1, 10))
        self.assertEqual(self.model.layers[2].db.shape, (1, 1))

        self.assertEqual(self.model.layers[2].Z.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].A.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].dZ.shape, (1, self.X.shape[1]))
        self.assertEqual(self.model.layers[2].dA.shape, (1, self.X.shape[1]))

        self.assertEqual(self.model.layers[1].W.shape, (10, 20))
        self.assertEqual(self.model.layers[1].b.shape, (10, 1))
        self.assertEqual(self.model.layers[1].dW.shape, (10, 20))
        self.assertEqual(self.model.layers[1].db.shape, (10, 1))

        self.assertEqual(self.model.layers[1].Z.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].A.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].dZ.shape, (10, self.X.shape[1]))
        self.assertEqual(self.model.layers[1].dA.shape, (10, self.X.shape[1]))

        self.assertEqual(self.model.layers[0].W.shape, (20, self.X.shape[0]))
        self.assertEqual(self.model.layers[0].b.shape, (20, 1))
        self.assertEqual(self.model.layers[0].dW.shape, (20, self.X.shape[0]))
        self.assertEqual(self.model.layers[0].db.shape, (20, 1))

        self.assertEqual(self.model.layers[0].Z.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].A.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].dZ.shape, (20, self.X.shape[1]))
        self.assertEqual(self.model.layers[0].dA.shape, (20, self.X.shape[1]))

        self.model.forward_prop(self.X)
        cost_1 = self.model.get_cost(self.Y)
        self.assertTrue(cost_1 - cost_0 > 0)
        self.assertEqual(cost_1, 8364284282.8210735)

    def test_backward_prop_with_update(self):
        pass
        # import pdb; pdb.set_trace()
        # self.model.add_layer(Layer(units=5, activation=relu, input_dim=self.X.shape[0]))
        # self.model.add_layer(Layer(units=1, activation=identity))
        # import pdb; pdb.set_trace()
        # self.model.fit(self.X, self.Y, 32)
        # print('---')
        # import pdb; pdb.set_trace()

    def test_optimization(self):
        pass

    def test_gradient_checking(self):
        pass
