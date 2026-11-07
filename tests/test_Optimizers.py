"""
Smoke tests for the L-BFGS-B optimization functions.

These run each optimizer for only a few iterations on simple, simulated data. They
check that the calls into scipy.optimize.fmin_l_bfgs_b are valid for the installed
version of scipy (scipy 1.18 removed the ``iprint`` argument, which broke these
functions), not that the optimizations converge.
"""

import os
import tempfile
import unittest

import demes
import numpy as np

import moments
import moments.LD


# A single deme with a size change, in which we fit the recent size.
GRAPH = """
time_units: generations
demes:
- name: A
  epochs:
  - {end_time: 200, start_size: 1000}
  - {end_time: 0, start_size: 2000}
"""

OPTIONS = """
parameters:
- name: N
  values:
  - demes:
      A:
        epochs:
          1: start_size
  lower_bound: 100
  upper_bound: 10000
"""


class SFSLBFGSBTestCase(unittest.TestCase):
    def setUp(self):
        self.model_func = moments.Demographics1D.two_epoch
        self.true_params = [2.0, 0.1]
        model = self.model_func(self.true_params, [20])
        self.data = moments.Spectrum(1000 * model.data, mask=model.mask)
        self.p0 = [1.5, 0.15]
        self.kwargs = dict(
            lower_bound=[1e-2, 1e-3], upper_bound=[10, 1], maxiter=3
        )

    def test_optimize_log_lbfgsb(self):
        popt = moments.Inference.optimize_log_lbfgsb(
            self.p0, self.data, self.model_func, **self.kwargs
        )
        self.assertEqual(len(popt), 2)
        self.assertTrue(np.all(np.isfinite(popt)))

    def test_optimize_lbfgsb(self):
        popt = moments.Inference.optimize_lbfgsb(
            self.p0, self.data, self.model_func, **self.kwargs
        )
        self.assertEqual(len(popt), 2)
        self.assertTrue(np.all(np.isfinite(popt)))


class LDLBFGSBTestCase(unittest.TestCase):
    def test_optimize_log_lbfgsb(self):
        model_func = moments.LD.Demographics1D.two_epoch
        rs = [0, 1e-6, 1e-5, 1e-4]
        Ne = 1000
        # expected normalized statistics, used as the data (the optimizer removes
        # the normalizing statistics itself)
        y = moments.LD.Inference.bin_stats(
            model_func, [2.0, 0.1], rho=4 * Ne * np.array(rs)
        )
        y = moments.LD.Inference.sigmaD2(y)
        means = [np.array(m) for m in y]
        varcovs = [np.diag(0.01 * np.abs(m) + 1e-6) for m in means]
        popt, LL = moments.LD.Inference.optimize_log_lbfgsb(
            [1.5, 0.15, Ne],
            [means, varcovs],
            [model_func],
            rs=rs,
            lower_bound=[1e-2, 1e-3, 100],
            upper_bound=[10, 1, 1e5],
            maxiter=3,
        )
        self.assertEqual(len(popt), 3)
        self.assertTrue(np.all(np.isfinite(popt)))


class DemesLBFGSBTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.graph_file = os.path.join(self.tmpdir.name, "graph.yaml")
        self.options_file = os.path.join(self.tmpdir.name, "options.yaml")
        with open(self.graph_file, "w") as fout:
            fout.write(GRAPH)
        with open(self.options_file, "w") as fout:
            fout.write(OPTIONS)
        self.g = demes.loads(GRAPH)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_optimize_sfs(self):
        fs = moments.Demes.SFS(self.g, samples={"A": 20}, theta=1000)
        param_names, popt, LL = moments.Demes.Inference.optimize(
            self.graph_file,
            self.options_file,
            fs,
            method="lbfgsb",
            maxiter=3,
        )
        self.assertEqual(param_names, ["N"])
        self.assertTrue(np.all(np.isfinite(popt)))

    def test_optimize_LD(self):
        rs = [0, 1e-6, 1e-5, 1e-4]
        y = moments.Demes.Inference.compute_bin_stats(self.g, ["A"], rs=rs)
        y = moments.LD.Inference.sigmaD2(y)
        y = moments.LD.Inference.remove_normalized_lds(y)
        means = [np.array(m) for m in y]
        varcovs = [np.diag(0.01 * np.abs(m) + 1e-6) for m in means]
        param_names, popt, LL = moments.Demes.Inference.optimize_LD(
            self.graph_file,
            self.options_file,
            means,
            varcovs,
            pop_ids=["A"],
            rs=rs,
            normalization="A",
            method="lbfgsb",
            maxiter=3,
        )
        self.assertEqual(param_names, ["N"])
        self.assertTrue(np.all(np.isfinite(popt)))


if __name__ == "__main__":
    unittest.main()
