import os
import unittest

import numpy as np
import scipy.special
import moments
import pickle
import time


class SpectrumTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_constant_migration_function(self):
        fs = moments.Demographics2D.snm([20, 20])
        fs.integrate([1, 1], 0.1, m=[[0, 1], [1, 0]])

        fs2 = moments.Demographics2D.snm([20, 20])
        mig_mat = lambda t: [[0, 1], [1, 0]]
        fs2.integrate([1, 1], 0.1, m=mig_mat)

        self.assertTrue(np.allclose(fs.data, fs2.data))

    def test_migration_function(self):
        fs = moments.Demographics2D.snm([20, 20])
        mig_mat = lambda t: [[0, 1 - 2 * t], [1 - 2 * t, 0]]
        fs.integrate([1, 1], 0.5, m=mig_mat)
        fs2 = moments.Demographics2D.snm([20, 20])
        fs2.integrate([1, 1], 0.5, m=[[0, 1], [1, 0]])
        self.assertTrue(np.all(fs.data > 0))
        self.assertTrue(fs.Fst() > fs2.Fst())

    def test_results(self):
        fs = moments.Demographics2D.snm([20, 20])
        fs.integrate([1, 1], 0.1, m=[[0, 3], [3, 0]])
        fs.integrate([1, 1], 0.1, m=[[0, 2], [2, 0]])
        fs.integrate([1, 1], 0.1, m=[[0, 0], [0, 0]])
        fs2 = moments.Demographics2D.snm([20, 20])

        def mig_mat(t):
            if 0 <= t <= 0.1:
                return [[0, 3], [3, 0]]
            elif 0.1 < t <= 0.2:
                return [[0, 2], [2, 0]]
            else:
                return [[0, 0], [0, 0]]

        fs2.integrate([1, 1], 0.3, m=mig_mat)
        self.assertTrue(np.allclose(fs.data, fs2.data, rtol=0.02, atol=1e-5))


suite = unittest.TestLoader().loadTestsFromTestCase(SpectrumTestCase)
