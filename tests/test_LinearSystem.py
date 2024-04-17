import os
import unittest
import numpy
import moments
import time
import sys

sys.path[:0] = ["../moments/"]

import Jackknife as jk
import LinearSystem_1D
import LinearSystem_2D


class LinearSystemTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_matrix_drift_1D(self):
        Dref = dlim = numpy.genfromtxt(
            os.path.join(os.path.dirname(__file__), "test_files/drift_matrix_1D.csv"),
            delimiter=",",
        )
        d = LinearSystem_1D.calcD(25).todense()
        self.assertTrue(numpy.allclose(d, Dref))

    def test_matrix_selection_1_1D(self):
        S1ref = dlim = numpy.genfromtxt(
            os.path.join(
                os.path.dirname(__file__), "test_files/selection_matrix_1_1D.csv"
            ),
            delimiter=",",
        )
        dims = numpy.array([25])
        ljk = jk.calcJK13(int(dims[0] - 1))
        s = LinearSystem_1D.calcS(dims[0], ljk)
        S = -0.1 * s
        self.assertTrue(numpy.allclose(S.todense(), S1ref))

    def test_matrix_selection_2_1D(self):
        S2ref = dlim = numpy.genfromtxt(
            os.path.join(
                os.path.dirname(__file__), "test_files/selection_matrix_2_1D.csv"
            ),
            delimiter=",",
        )
        dims = numpy.array([25])
        ljk = jk.calcJK23(int(dims[0] - 1))
        s = LinearSystem_1D.calcS2(dims[0], ljk)
        S = -1.0 * (1 - 2 * 0.1) * s
        self.assertTrue(numpy.allclose(S.todense(), S2ref))

    def test_matrix_drift_2D(self):
        Dref = dlim = numpy.genfromtxt(
            os.path.join(os.path.dirname(__file__), "test_files/drift_matrix.csv"),
            delimiter=",",
        )
        dims = numpy.array([25, 30])
        d1 = LinearSystem_2D.calcD1(dims)
        d2 = LinearSystem_2D.calcD2(dims)
        d = d1.todense() + d2.todense()
        self.assertTrue(numpy.allclose(d, Dref))

    def test_matrix_selection_1_2D(self):
        S1ref = dlim = numpy.genfromtxt(
            os.path.join(
                os.path.dirname(__file__), "test_files/selection_matrix_1.csv"
            ),
            delimiter=",",
        )
        dims = numpy.array([25, 30])
        ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(len(dims))]
        s1 = LinearSystem_2D.calcS_1(dims, ljk[0])
        s2 = LinearSystem_2D.calcS_2(dims, ljk[1])
        S1 = -0.1 * s1 + 0.4 * s2
        self.assertTrue(numpy.allclose(S1.todense(), S1ref))

    def test_matrix_selection_2_2D(self):
        S2ref = dlim = numpy.genfromtxt(
            os.path.join(
                os.path.dirname(__file__), "test_files/selection_matrix_2.csv"
            ),
            delimiter=",",
        )
        dims = numpy.array([25, 30])
        ljk = [jk.calcJK23(int(dims[i] - 1)) for i in range(len(dims))]
        s1 = LinearSystem_2D.calcS2_1(dims, ljk[0])
        s2 = LinearSystem_2D.calcS2_2(dims, ljk[1])
        S2 = -0.8 * s1 + 0.2 * s2
        self.assertTrue(numpy.allclose(S2.todense(), S2ref))

    def test_matrix_migration_2D(self):
        Mref = dlim = numpy.genfromtxt(
            os.path.join(os.path.dirname(__file__), "test_files/migration_matrix.csv"),
            delimiter=",",
        )
        dims = numpy.array([25, 30])
        m = numpy.array([[1, 5], [10, 1]])
        ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(len(dims))]
        m1 = LinearSystem_2D.calcM_1(dims, ljk[1])
        m2 = LinearSystem_2D.calcM_2(dims, ljk[0])
        M = m[0, 1] * m1 + m[1, 0] * m2

        self.assertTrue(numpy.allclose(M.todense(), Mref))

    def test_steady_state(self):
        # test that integrating from steady_state doesn't make us leave steady state
        for gamma in [-10, 1, 10]:
            for h in [-0.1, 0.2, 1]:
                n = 100
                steady = moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma, h=h)
                after = moments.Integration_nomig.integrate_nomig(
                    steady, Npop=[1], tf=1, gamma=gamma, h=h
                )
        self.assertTrue(numpy.allclose(steady, after))


suite = unittest.TestLoader().loadTestsFromTestCase(LinearSystemTestCase)
if __name__ == "__main__":
    unittest.main()
