import os
import unittest
import numpy as np
import time

from moments import Jackknife as jk
from moments import LinearSystem_1D as ls1
import moments


class ConservativeSelectionMatrix(unittest.TestCase):
    # Make sure that transition matrices do not add or remove density
    # Note that this does not ensure that the result is _correct_
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_additive_selection_matrix(self):
        n = np.array([10, 20, 25])
        num_pops = len(n)
        dims = np.array(n + np.ones(num_pops), dtype=int)
        ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(num_pops)]
        vs = [ls1.calcS(dims[i], ljk[i]) for i in range(num_pops)]
        for v in vs:
            self.assertTrue(np.allclose(v.toarray().sum(0), 0))

    def test_dominance_selection_matrix(self):
        n = np.array([10, 20, 25])
        num_pops = len(n)
        dims = np.array(n + np.ones(num_pops), dtype=int)
        ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(num_pops)]
        vs2 = [ls1.calcS2(dims[i], ljk2[i]) for i in range(num_pops)]
        for v in vs2:
            self.assertTrue(np.allclose(v.toarray().sum(0), 0))

    def test_underdominance_selection_matrix(self):
        n = np.array([10, 20, 25])
        num_pops = len(n)
        dims = np.array(n + np.ones(num_pops), dtype=int)
        ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(num_pops)]
        vs3 = [ls1.calcUnderdominance(dims[i], ljk2[i]) for i in range(num_pops)]
        for v in vs3:
            self.assertTrue(np.allclose(v.toarray().sum(0), 0))

    def test_additive_with_migration(self):
        n = np.array([10, 20, 25])
        num_pops = len(n)
        dims = np.array(n + np.ones(num_pops), dtype=int)
        ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(num_pops)]
        vs = moments.Integration._calcS(dims, ljk)
        S1 = moments.Integration._buildS(vs, dims, [0.1, 2, 7.3], [0.5, 1, 0.15])
        for S in S1:
            self.assertTrue(np.allclose(S.sum(0), 0))

    def test_dominance_with_migration(self):
        n = np.array([10, 20, 25])
        num_pops = len(n)
        dims = np.array(n + np.ones(num_pops), dtype=int)
        ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(num_pops)]
        vs2 = moments.Integration._calcS2(dims, ljk2)
        S2 = moments.Integration._buildS2(vs2, dims, [0.1, 2, 7.3], [0.5, 1, 0.15])
        for S in S2:
            self.assertTrue(np.allclose(S.sum(0), 0))

    def test_underdominance_with_migratin(self):
        n = np.array([10, 20, 25])
        num_pops = len(n)
        dims = np.array(n + np.ones(num_pops), dtype=int)
        ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(num_pops)]
        vs3 = moments.Integration._calcUnderdominance(dims, ljk2)
        S3 = moments.Integration._buildS3(vs3, dims, [0.1, 2, 7.3], [0.5, 1, 0.15])
        for S in S3:
            self.assertTrue(np.allclose(S.sum(0), 0))


class AdditiveVsDominance(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_nearly_additive(self):
        n = 40
        gamma = -5
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(40, gamma=-5))
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(40, gamma=-5))
        fs.integrate([1], 0.5, gamma=gamma)
        fs2.integrate([1], 0.5, gamma=gamma, h=0.5 - 1e-4)
        self.assertTrue(np.allclose(fs, fs2, rtol=0.0005))


class Overdominance(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_steady_state(self):
        n = 40
        for gamma, h, overdominance in [
            [0, 0.5, -2],
            [0, 0.5, -10],
            [0, 0.5, 5],
            [-1, 0.5, -2],
            [2, 0.1, -5],
        ]:
            fs = moments.Spectrum(
                moments.LinearSystem_1D.steady_state_1D(
                    n, gamma=gamma, h=h, overdominance=overdominance
                )
            )
            fs2 = fs.copy()
            fs.integrate([1], 0.5, gamma=gamma, h=h, overdominance=overdominance)
            self.assertTrue(np.allclose(fs, fs2))

    def test_underdominance(self):
        n = 100
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=-1, h=4))
        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, gamma=-1, h=4)
        )
        fs.integrate([1], 1, gamma=-1, h=4)
        fs2.integrate([1], 1, gamma=-1, overdominance=-7)
        self.assertTrue(np.allclose(fs, fs2, rtol=0.02))

    def test_overdominance(self):
        n = 100
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=2, h=4))
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=2, h=4))
        fs.integrate([1], 1, gamma=2, h=4)
        fs2.integrate([1], 1, gamma=2, overdominance=14)
        self.assertTrue(np.allclose(fs, fs2, rtol=0.01))

    def test_two_pop_nomig(self):
        n = 50
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, overdominance=-4)
        )
        fs0 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, overdominance=-4)
        )
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], 0.5, overdominance=-4)
        self.assertTrue(np.allclose(fs0, fs.marginalize([0]), rtol=0.005))
        self.assertTrue(np.allclose(fs0, fs.marginalize([1]), rtol=0.005))

    def test_two_pop_nomig_size_change(self):
        n = 50
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, overdominance=-4)
        )
        fs0 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, overdominance=-4)
        )
        fs1 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, overdominance=-4)
        )
        fs = fs.split(0, n, n)
        fs.integrate([2, 0.5], 0.5, overdominance=-4)
        fs0.integrate([2], 0.5, overdominance=-4)
        fs1.integrate([0.5], 0.5, overdominance=-4)
        self.assertTrue(np.allclose(fs0, fs.marginalize([1]), rtol=0.005))
        self.assertTrue(np.allclose(fs1, fs.marginalize([0]), rtol=0.005))

    def test_two_pop_with_negligible_migration(self):
        n = 50
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, overdominance=-1)
        )
        fs0 = fs.project([n])
        fs = fs.split(0, n, n)
        m = [[0, 1e-6], [0, 0]]
        fs.integrate([1, 1], 0.2, m=m, overdominance=-1)
        self.assertTrue(np.allclose(fs0, fs.marginalize([0]), rtol=0.001))
        self.assertTrue(np.allclose(fs0, fs.marginalize([1]), rtol=0.001))
    
    def test_two_pop_with_migration(self):
        n = 50
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, overdominance=-1)
        )
        fs0 = fs.project([n])
        fs = fs.split(0, n, n)
        m = [[0, 1], [0, 0]]
        fs.integrate([1, 1], 0.2, m=m, overdominance=-1)
        self.assertTrue(np.allclose(fs0, fs.marginalize([0]), rtol=0.001))
        self.assertFalse(np.allclose(fs0, fs.marginalize([1]), rtol=0.001))
