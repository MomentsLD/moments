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
        # tests that having pop_ids on a spectrum doesn't cause an error when splitting
        fs = moments.Demographics1D.snm([20])
        fs.pop_ids = ["pop0"]
        fs = moments.Manips.split_1D_to_2D(fs, 10, 10)
        self.assertTrue(fs.shape == (11, 11))
        self.assertTrue(fs.pop_ids == None)

        fs.pop_ids = ["pop1", "pop2"]
        fs1 = moments.Manips.split_2D_to_3D_1(fs, 5, 5)
        fs2 = moments.Manips.split_2D_to_3D_2(fs, 5, 5)
        self.assertTrue(fs1.pop_ids == None)
        self.assertTrue(fs2.pop_ids == None)
        self.assertTrue(fs1.shape == (6, 11, 6))
        self.assertTrue(fs2.shape == (11, 6, 6))

        fs2.pop_ids = ["A", "B", "C"]
        fs3 = moments.Manips.split_3D_to_4D_3(fs2, 1, 4)
        self.assertTrue(fs3.pop_ids == None)
        self.assertTrue(fs3.shape == (11, 6, 2, 5))

        fs3.pop_ids = ["A", "B", "C", "D"]
        fs4 = moments.Manips.split_4D_to_5D_4(fs3, 2, 2)
        self.assertTrue(fs4.pop_ids == None)
        self.assertTrue(fs4.shape == (11, 6, 2, 3, 3))

    def test_swap_axes_pop_ids(self):
        fs = moments.Demographics2D.snm([10, 10])
        fs.pop_ids = ["A", "B"]
        self.assertTrue(fs.pop_ids == ["A", "B"])
        fs = fs.swap_axes(0, 1)
        self.assertTrue(fs.pop_ids == ["B", "A"])

    def test_split_projection_1D_to_2D(self):
        fs = moments.Demographics1D.snm([20])
        fs1 = moments.Manips.split_1D_to_2D(fs, 5, 10)
        fs2 = moments.Manips.split_1D_to_2D(fs, 10, 10)
        fs2 = fs2.project([5, 10])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

        fs = moments.Demographics1D.snm([20])
        fs1 = moments.Manips.split_1D_to_2D(fs, 10, 6)
        fs2 = moments.Manips.split_1D_to_2D(fs, 10, 10)
        fs2 = fs2.project([10, 6])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

        fs = moments.Demographics1D.snm([20])
        fs1 = moments.Manips.split_1D_to_2D(fs, 3, 6)
        fs2 = moments.Manips.split_1D_to_2D(fs, 10, 10)
        fs2 = fs2.project([3, 6])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

    def test_split_projection_2D_to_3D(self):
        fs = moments.Demographics2D.snm([10, 10])

        fs1 = moments.Manips.split_2D_to_3D_1(fs, 3, 4)
        fs2 = moments.Manips.split_2D_to_3D_1(fs, 5, 5)
        fs2 = fs2.project([3, 10, 4])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

        fs1 = moments.Manips.split_2D_to_3D_2(fs, 3, 4)
        fs2 = moments.Manips.split_2D_to_3D_2(fs, 5, 5)
        fs2 = fs2.project([10, 3, 4])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

    def test_split_projection_3D_to_4D(self):
        fs = moments.Demographics2D.snm([10, 20])
        fs = moments.Manips.split_2D_to_3D_2(fs, 10, 10)

        fs1 = moments.Manips.split_3D_to_4D_3(fs, 3, 4)
        fs2 = moments.Manips.split_3D_to_4D_3(fs, 5, 5)
        fs2 = fs2.project([10, 10, 3, 4])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

    def test_split_4D_to_5D_3_dimension(self):
        fs = moments.Demographics2D.snm([10, 30])
        fs = moments.Manips.split_2D_to_3D_2(fs, 10, 20)
        fs = moments.Manips.split_3D_to_4D_3(fs, 10, 10)
        fs = moments.Manips.split_4D_to_5D_3(fs, 5, 5)
        self.assertTrue(numpy.all(fs.sample_sizes == [10, 10, 5, 10, 5]))

    def test_split_projection_4D_to_5D(self):
        fs = moments.Demographics2D.snm([10, 30])
        fs = moments.Manips.split_2D_to_3D_2(fs, 10, 20)
        fs = moments.Manips.split_3D_to_4D_3(fs, 10, 10)

        fs1 = moments.Manips.split_4D_to_5D_3(fs, 3, 4)
        fs2 = moments.Manips.split_4D_to_5D_3(fs, 5, 5)
        fs2 = fs2.project([10, 10, 3, 10, 4])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

        fs1 = moments.Manips.split_4D_to_5D_4(fs, 3, 4)
        fs2 = moments.Manips.split_4D_to_5D_4(fs, 5, 5)
        fs2 = fs2.project([10, 10, 10, 3, 4])
        self.assertTrue(numpy.allclose(fs1.data, fs2.data))

    def test_split_4D_3_vs_4(self):
        fs = moments.Demographics2D.snm([10, 30])
        fs = moments.Manips.split_2D_to_3D_2(fs, 10, 20)
        fs = moments.Manips.split_3D_to_4D_3(fs, 10, 10)
        self.assertTrue(numpy.all(fs.data == fs.swapaxes(2, 3).data))

        fs1 = moments.Manips.split_4D_to_5D_3(fs, 5, 5)
        fs2 = moments.Manips.split_4D_to_5D_4(fs, 5, 5)
        fs1 = fs1.swapaxes(2, 3)

        self.assertTrue(numpy.all(fs1.sample_sizes == fs2.sample_sizes))
        self.assertTrue(numpy.all(fs1.data == fs2.data))

    def test_split_3D_to_4D_1(self):
        fs = moments.Spectrum(numpy.random.rand(11 * 7 * 9).reshape((11, 7, 9)))
        fs = moments.Manips.split_3D_to_4D_1(fs, 3, 7)
        self.assertTrue(numpy.all(fs.sample_sizes == [3, 6, 8, 7]))

    def test_split_3D_to_4D_2(self):
        fs = moments.Spectrum(numpy.random.rand(7 * 11 * 9).reshape((7, 11, 9)))
        fs = moments.Manips.split_3D_to_4D_2(fs, 3, 7)
        self.assertTrue(numpy.all(fs.sample_sizes == [6, 3, 8, 7]))

    def test_split_4D_to_5D_1(self):
        fs = moments.Spectrum(numpy.random.rand(11 * 7 * 9 * 5).reshape((11, 7, 9, 5)))
        fs = moments.Manips.split_4D_to_5D_1(fs, 3, 7)
        self.assertTrue(numpy.all(fs.sample_sizes == [3, 6, 8, 4, 7]))

    def test_split_4D_to_5D_2(self):
        fs = moments.Spectrum(numpy.random.rand(11 * 7 * 9 * 5).reshape((7, 11, 9, 5)))
        fs = moments.Manips.split_4D_to_5D_2(fs, 3, 7)
        self.assertTrue(numpy.all(fs.sample_sizes == [6, 3, 8, 4, 7]))


suite = unittest.TestLoader().loadTestsFromTestCase(ManipsTestCase)
