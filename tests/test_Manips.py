import os
import unittest

import numpy
import scipy.special
import moments
import pickle
import time

class ManipsTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_split_with_pop_ids(self):
        fs = moments.Demographics1D.snm([20])
        fs.pop_ids = ['pop0']
        fs = moments.Manips.split_1D_to_2D(fs, 10, 10)
        self.assertTrue(fs.shape == (11,11))
        self.assertTrue(fs.pop_ids == None)
        fs.pop_ids = ['pop1','pop2']
        fs1 = moments.Manips.split_2D_to_3D_1(fs, 5, 5)
        fs2 = moments.Manips.split_2D_to_3D_2(fs, 5, 5)
        self.assertTrue(fs1.pop_ids == None)
        self.assertTrue(fs2.pop_ids == None)
        self.assertTrue(fs1.shape == (6,11,6))
        self.assertTrue(fs2.shape == (11,6,6))
        fs2.pop_ids = ['A','B','C']
        fs3 = moments.Manips.split_3D_to_4D_3(fs2, 1, 4)
        self.assertTrue(fs3.pop_ids == None)
        self.assertTrue(fs3.shape == (11,6,2,5))
        fs3.pop_ids = ['A','B','C','D']
        fs4 = moments.Manips.split_4D_to_5D_4(fs3, 2, 2)
        self.assertTrue(fs4.pop_ids == None)
        self.assertTrue(fs4.shape == (11,6,2,3,3))


suite = unittest.TestLoader().loadTestsFromTestCase(ManipsTestCase)
