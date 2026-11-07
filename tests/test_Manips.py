import os
import unittest
import unittest.mock

import numpy
import scipy.optimize
import scipy.special
import scipy.stats
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


class NNLSTestCase(unittest.TestCase):
    """
    Tests of the non-negative least squares solver used by admix_inplace
    (issue #147).
    """

    def admix_problem(self, n_draws, n_lineages, m1):
        # the nnls problem that admix_inplace solves to approximate the admixture
        gamma = moments.Manips.__Gamma__(n_draws, n_lineages)
        target = scipy.stats.binom(n_lineages, m1).pmf(numpy.arange(n_lineages + 1))
        return gamma.transpose(), target

    def test_matches_scipy_nnls(self):
        # On an easy admixture problem (converges with scipy's defaults), our
        # wrapper gives the same solution and residual as scipy's nnls.
        A, b = self.admix_problem(10, 10, 0.1)
        x, rnorm = moments.Manips.__nnls_mod__(A, b)
        x_scipy, rnorm_scipy = scipy.optimize.nnls(A, b)
        self.assertTrue(numpy.allclose(x, x_scipy))
        self.assertTrue(numpy.isclose(rnorm, rnorm_scipy))

    def test_converges_beyond_default_maxiter(self):
        # A problem where scipy >= 1.15's nnls fails at its default maxiter (3 * n):
        # our larger maxiter solves it, giving a non-negative x and no warning.
        A, b = self.admix_problem(29, 29, 0.01)
        # pytest is configured to turn warnings into errors, so this also checks
        # that we don't need to fall back to lsq_linear
        x, rnorm = moments.Manips.__nnls_mod__(A, b)
        self.assertTrue(numpy.all(x >= 0))

    def test_fallback_when_nnls_fails(self):
        # Force nnls to fail (mock it to raise): we warn, fall back to lsq_linear,
        # and get a non-negative x with the same residual as the converged nnls.
        A, b = self.admix_problem(29, 29, 0.01)
        x_nnls, rnorm_nnls = moments.Manips.__nnls_mod__(A, b)
        with unittest.mock.patch(
            "moments.Manips.nnls", side_effect=RuntimeError("did not converge")
        ):
            with self.assertWarns(UserWarning):
                x, rnorm = moments.Manips.__nnls_mod__(A, b)
        self.assertTrue(numpy.all(x >= 0))
        self.assertTrue(numpy.isclose(rnorm, rnorm_nnls, atol=1e-5))
        self.assertTrue(numpy.isclose(rnorm, numpy.linalg.norm(A @ x - b)))

    def test_pulse_migrate(self):
        # Regression test: this pulse migration used to crash with "RuntimeError:
        # Maximum number of iterations reached". Now runs and gives a finite SFS.
        fs = moments.Demographics2D.snm([58, 29])
        fs_pulse = fs.pulse_migrate(0, 1, 29, 0.01)
        self.assertEqual(fs_pulse.sample_sizes.tolist(), [29, 29])
        self.assertTrue(numpy.all(numpy.isfinite(fs_pulse)))
