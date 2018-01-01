import numpy as np
import unittest as ut

from ..costs import MSE


class TestGraphModel(ut.TestCase):
    def test_MSE(self):
        AL = np.array([1, 2, 3, 4])
        Y0 = np.array([1, 2, 3, 4])  # For 0
        Y1 = np.array([1, 1, 1, 1])  # For (0 + 1 + 2**2 + 3**2)/4 = 14/4 = 3.5
        cost0 = MSE(AL, Y0)
        self.assertEqual(cost0, 0)

        cost1 = MSE(AL, Y1)
        self.assertEqual(cost1, 3.5)
