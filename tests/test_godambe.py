import unittest
import warnings
from unittest import mock

import numpy as np
import moments
from moments.Godambe import _check_nested_hessian


class CheckNestedHessian(unittest.TestCase):
    """
    The warning in LRT_adjust for an indefinite or nearly singular Hessian of the
    complex model at the best fit of the nested model.
    """

    labels = ["a", "b", "c"]

    def check(self, H):
        # Runs the check, returning whether it warned and the warning messages.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warned = _check_nested_hessian(H, self.labels, "test")
        return warned, [str(x.message) for x in w]

    def test_well_conditioned(self):
        # A positive definite Hessian with moderate correlations and very different
        # parameter scales does not warn.
        C = np.array([[1, 0.5, 0.2], [0.5, 1, -0.3], [0.2, -0.3, 1]])
        s = np.array([1e-4, 1.0, 1e6])
        warned, msgs = self.check(C * np.outer(s, s))
        self.assertFalse(warned)
        self.assertEqual(msgs, [])

    def test_nearly_singular(self):
        # Parameters b and c are almost perfectly correlated, so the correlation-
        # scaled Hessian has a tiny eigenvalue along b and c, whatever the scales.
        C = np.array([[1, 0.1, 0.1], [0.1, 1, 0.999], [0.1, 0.999, 1]])
        s = np.array([1.0, 1e-3, 1e4])
        warned, msgs = self.check(C * np.outer(s, s))
        self.assertTrue(warned)
        self.assertEqual(len(msgs), 1)
        self.assertIn("nearly singular", msgs[0])
        self.assertIn("mostly in b, c", msgs[0])

    def test_indefinite(self):
        # A negative eigenvalue along a and b is reported as indefinite.
        H = np.array([[1, 1.2, 0], [1.2, 1, 0], [0, 0, 1]])
        warned, msgs = self.check(H)
        self.assertTrue(warned)
        self.assertIn("indefinite", msgs[0])
        self.assertIn("a, b", msgs[0])

    def test_nonpositive_diagonal(self):
        # A non-positive diagonal entry is reported for that parameter, without
        # trying to scale by its square root.
        H = np.diag([1.0, -2.0, 3.0])
        warned, msgs = self.check(H)
        self.assertTrue(warned)
        self.assertIn("not concave in b", msgs[0])

    def test_nonfinite(self):
        # Non-finite entries warn instead of raising from the eigendecomposition.
        H = np.eye(3)
        H[0, 1] = H[1, 0] = np.nan
        warned, msgs = self.check(H)
        self.assertTrue(warned)
        self.assertIn("non-finite", msgs[0])


class SFSLRTAdjustHessianCheck(unittest.TestCase):
    """
    moments.Godambe.LRT_adjust checks the Hessian over all parameters of the complex
    model, including theta, without changing the adjustment factor.
    """

    def setUp(self):
        # Poisson-sampled data from a split model without migration, with
        # Poisson-resampled bootstrap replicates, testing for migration (index 3).
        np.random.seed(42)
        self.func = moments.Demographics2D.split_mig
        self.p0 = [2.0, 3.0, 0.2, 0.0]
        model = 10000 * self.func(self.p0, [10, 10])
        self.data = model.sample()
        self.boots = [self.data.sample() for _ in range(20)]

    def test_no_warning_and_same_adjustment(self):
        # The data fit the nested model, so the check doesn't warn, and the
        # adjustment factor is the same with and without the check.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adj = moments.Godambe.LRT_adjust(
                self.func, self.boots, self.p0, self.data, [3]
            )
        self.assertEqual([x for x in w if "Hessian" in str(x.message)], [])
        adj_nocheck = moments.Godambe.LRT_adjust(
            self.func, self.boots, self.p0, self.data, [3], check_hessian=False
        )
        self.assertTrue(np.isfinite(adj) and adj > 0)
        self.assertEqual(adj, adj_nocheck)

    def test_checks_full_hessian(self):
        # With check_hessian=True, the check gets the Hessian over all parameters
        # plus theta, and with check_hessian=False, only the nested submatrix.
        with mock.patch("moments.Godambe._check_nested_hessian") as check:
            moments.Godambe.LRT_adjust(self.func, self.boots, self.p0, self.data, [3])
            H, labels, _ = check.call_args.args
            self.assertEqual(H.shape, (5, 5))
            self.assertEqual(labels, ["p0[0]", "p0[1]", "p0[2]", "p0[3]", "theta"])
            moments.Godambe.LRT_adjust(
                self.func, self.boots, self.p0, self.data, [3], check_hessian=False
            )
            H, labels, _ = check.call_args.args
            self.assertEqual(H.shape, (1, 1))
            self.assertEqual(labels, ["p0[3]"])


if __name__ == "__main__":
    unittest.main()
