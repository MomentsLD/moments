"""
Tests for moments.Demes.Inference.
"""

import os
import tempfile
import unittest
import unittest.mock

import demes
import numpy as np

import moments
from moments.Demes import Inference

GRAPH = """
time_units: generations
demes:
- name: anc
  epochs:
  - {end_time: 1000, start_size: 1000}
- name: A
  ancestors: [anc]
  epochs:
  - {end_time: 0, start_size: 2000}
- name: B
  ancestors: [anc]
  epochs:
  - {end_time: 0, start_size: 3000}
migrations:
- {demes: [A, B], rate: 1e-4}
pulses:
- {sources: [A], dest: B, time: 100, proportions: [0.1]}
"""


# the options "values" entry for the start size of deme A
SIZE_A = [{"demes": {"A": {"epochs": {0: "start_size"}}}}]


def size_param(name, deme, epoch=0, attribute="start_size"):
    """An options file parameter entry for a deme's epoch size."""
    return f"""
- name: {name}
  values:
  - demes:
      {deme}:
        epochs:
          {epoch}: {attribute}
"""


class TempFilesTestCase(unittest.TestCase):
    """Base class for tests that write graph and options YAML files."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def write(self, fname, contents):
        path = os.path.join(self.tmpdir.name, fname)
        with open(path, "w") as fout:
            fout.write(contents)
        return path


class TestDuplicateParameterValues(TempFilesTestCase):
    """
    Two parameters mapping to the same value in the graph should raise an error
    (issue #235).
    """

    def load_options(self, params):
        return Inference._get_params_dict(
            self.write("options.yaml", "parameters:" + params)
        )

    def test_distinct_values(self):
        # Params on two different demes' sizes: loads fine, both params kept.
        options = self.load_options(size_param("NA", "A") + size_param("NB", "B"))
        self.assertEqual([p["name"] for p in options["parameters"]], ["NA", "NB"])

    def test_one_parameter_maps_to_several_values(self):
        # One param setting the sizes of both A and B (a shared param) is fine.
        params = """
- name: N
  values:
  - demes:
      A:
        epochs:
          0: start_size
      B:
        epochs:
          0: start_size
"""
        self.load_options(params)

    def test_same_value_repeated_in_one_parameter(self):
        # One param listing A's size twice is redundant but harmless, so no error.
        params = """
- name: N
  values:
  - demes:
      A:
        epochs:
          0: start_size
  - demes:
      A:
        epochs:
          0: start_size
"""
        self.load_options(params)

    def test_duplicate_epoch_value(self):
        # Two params both set A's start_size: raises, and the error names both.
        with self.assertRaises(ValueError) as err:
            self.load_options(size_param("N1", "A") + size_param("N2", "A"))
        self.assertIn("N1", str(err.exception))
        self.assertIn("N2", str(err.exception))

    def test_duplicate_within_multiple_values(self):
        # N2 sets the sizes of B and A, so it overlaps with N1 on A: raises.
        params = size_param("N1", "A") + """
- name: N2
  values:
  - demes:
      B:
        epochs:
          0: start_size
      A:
        epochs:
          0: start_size
"""
        with self.assertRaises(ValueError):
            self.load_options(params)

    def test_different_attributes_of_same_epoch(self):
        # start_size and end_size of one epoch are different values: no error.
        params = size_param("N0", "A") + size_param("NF", "A", attribute="end_size")
        self.load_options(params)

    def test_duplicate_migration_rate(self):
        # Two params on the same migration rate: raises.
        params = """
- name: m1
  values:
  - migrations:
      0: rate
- name: m2
  values:
  - migrations:
      0: rate
"""
        with self.assertRaises(ValueError):
            self.load_options(params)

    def test_duplicate_pulse_values(self):
        # Two params on the same pulse time, or the same pulse proportion, raise.
        # The time and proportion of one pulse are different values, so are fine.
        pulse_time = """
- name: {}
  values:
  - pulses:
      0: time
"""
        with self.assertRaises(ValueError):
            self.load_options(pulse_time.format("T1") + pulse_time.format("T2"))
        pulse_prop = """
- name: {}
  values:
  - pulses:
      0:
        proportions: 0
"""
        with self.assertRaises(ValueError):
            self.load_options(pulse_prop.format("f1") + pulse_prop.format("f2"))
        # pulse time and proportion are different values
        self.load_options(pulse_time.format("T") + pulse_prop.format("f"))

    def test_duplicate_deme_start_time(self):
        # Two params on deme A's start_time: raises.
        start_time = """
- name: {}
  values:
  - demes:
      A: start_time
"""
        with self.assertRaises(ValueError):
            self.load_options(start_time.format("T1") + start_time.format("T2"))

    def test_optimize_raises(self):
        # End to end: optimize() raises on duplicate params when it loads the options,
        # before any fitting happens.
        graph = self.write("graph.yaml", GRAPH)
        options = self.write(
            "options.yaml",
            "parameters:" + size_param("N1", "A") + size_param("N2", "A"),
        )
        fs = moments.Demes.SFS(demes.loads(GRAPH), samples={"A": 4, "B": 4})
        with self.assertRaises(ValueError):
            Inference.optimize(graph, options, fs, maxiter=1)


class TestOptionsValidation(TempFilesTestCase):
    """Errors for invalid options files, bounds, and constraints."""

    def setUp(self):
        super().setUp()
        self.builder = demes.loads_asdict(GRAPH)

    def load_options(self, contents):
        return Inference._get_params_dict(self.write("options.yaml", contents))

    def test_parameters_required(self):
        # Options file without a `parameters` section: raises.
        with self.assertRaises(ValueError):
            self.load_options("constraints: []\n")

    def test_default_bounds(self):
        # No bounds given: lower defaults to 0 and upper to inf, and the initial
        # value (2000) is read from the graph.
        options = self.load_options("parameters:" + size_param("NA", "A"))
        names, p0, lower, upper = Inference._set_up_params_and_bounds(
            options, self.builder
        )
        self.assertEqual(names, ["NA"])
        self.assertEqual(p0[0], 2000)
        self.assertEqual(lower[0], 0)
        self.assertEqual(upper[0], np.inf)

    def test_invalid_bounds(self):
        # Each bad bounds setup raises: lower > upper, or the initial value outside
        # of or on a bound.
        for bounds in [
            "  lower_bound: 5000\n  upper_bound: 100\n",  # lower > upper
            "  lower_bound: 3000\n",  # initial value below lower bound
            "  upper_bound: 1000\n",  # initial value above upper bound
            "  lower_bound: 2000\n",  # initial value equal to the bound
        ]:
            options = self.load_options("parameters:" + size_param("NA", "A") + bounds)
            with self.assertRaises(ValueError):
                Inference._set_up_params_and_bounds(options, self.builder)

    def test_value_not_in_graph(self):
        # Params pointing at a deme (C) or epoch (A's epoch 1) that doesn't exist raise.
        for params in [size_param("NC", "C"), size_param("NA", "A", epoch=1)]:
            options = self.load_options("parameters:" + params)
            with self.assertRaises(ValueError):
                Inference._set_up_params_and_bounds(options, self.builder)

    def test_constraints(self):
        # Constraint NB > NA: cons(x) is positive when satisfied and negative when
        # violated. Without constraints, we get None.
        params = size_param("NA", "A") + size_param("NB", "B")
        options = self.load_options(
            "parameters:"
            + params
            + "constraints:\n- params: [NB, NA]\n  constraint: greater_than\n"
        )
        cons = Inference._set_up_constraints(options, ["NA", "NB"])
        # constraints are satisfied when all values are positive
        self.assertTrue(np.all(cons(np.array([2000, 3000])) > 0))
        self.assertTrue(np.all(cons(np.array([3000, 2000])) < 0))
        self.assertIsNone(Inference._set_up_constraints({"parameters": []}, []))

    def test_invalid_constraints(self):
        # Each raises: unknown constraint type, three params, or an unknown param name.
        params = size_param("NA", "A") + size_param("NB", "B")
        for cons in [
            "- params: [NB, NA]\n  constraint: bigger_than\n",
            "- params: [NB, NA, NA]\n  constraint: greater_than\n",
            "- params: [NB, NC]\n  constraint: greater_than\n",
        ]:
            options = self.load_options(
                "parameters:" + params + "constraints:\n" + cons
            )
            with self.assertRaises(ValueError):
                Inference._set_up_constraints(options, ["NA", "NB"])

    def test_perturb_respects_bounds_and_constraints(self):
        # 50 random (seeded) perturbations all stay within bounds and satisfy NB > NA.
        np.random.seed(1)
        p0 = np.array([2000.0, 3000.0])
        lower = np.array([1500.0, 1500.0])
        upper = np.array([2500.0, 4000.0])
        cons = lambda x: np.array([x[1] - x[0]])
        for _ in range(50):
            p = Inference._perturb_params_constrained(p0, 1, lower, upper, cons)
            self.assertTrue(np.all(p > lower) and np.all(p < upper))
            self.assertTrue(p[1] > p[0])


class TestSetValues(unittest.TestCase):
    """Updating graph values from parameters."""

    def test_shared_parameter_sets_all_values(self):
        # Setting a param that maps to the sizes of A and B updates both.
        builder = demes.loads_asdict(GRAPH)
        values = [
            {"demes": {"A": {"epochs": {0: "start_size"}}}},
            {"demes": {"B": {"epochs": {0: "start_size"}}}},
        ]
        builder = Inference._set_value(builder, values, 1234)
        g = demes.Graph.fromdict(builder)
        self.assertEqual(g["A"].epochs[0].start_size, 1234)
        self.assertEqual(g["B"].epochs[0].start_size, 1234)

    def test_ancestry_proportions(self):
        # Get and set an ancestry proportion of admixed deme C, using a 3-deme graph.
        builder = demes.loads_asdict("""
time_units: generations
demes:
- name: A
  epochs: [{end_time: 100, start_size: 1000}]
- name: B
  epochs: [{end_time: 100, start_size: 1000}]
- name: C
  ancestors: [A, B]
  proportions: [0.3, 0.7]
  start_time: 100
  epochs: [{end_time: 0, start_size: 1000}]
""")
        values = [{"demes": {"C": {"proportions": 0}}}]
        self.assertEqual(Inference._get_value(builder, values), 0.3)
        # setting a proportion updates the last proportion so they sum to one
        builder = Inference._set_value(builder, values, 0.4)
        self.assertTrue(np.allclose(builder["demes"][2]["proportions"], [0.4, 0.6]))
        # the last proportion is determined by the others, so can't be set
        with self.assertRaises(ValueError):
            Inference._set_value(builder, [{"demes": {"C": {"proportions": 1}}}], 0.5)
        with self.assertRaises(ValueError):
            Inference._set_value(builder, values, 1.5)

    def test_migration_and_pulse_values(self):
        # Set a migration rate, pulse proportion, and pulse time, and read them back.
        builder = demes.loads_asdict(GRAPH)
        values = [{"migrations": {0: "rate"}}]
        builder = Inference._set_value(builder, values, 2e-4)
        self.assertEqual(Inference._get_value(builder, values), 2e-4)
        values = [{"pulses": {0: {"proportions": 0}}}]
        builder = Inference._set_value(builder, values, 0.2)
        self.assertEqual(Inference._get_value(builder, values), 0.2)
        values = [{"pulses": {0: "time"}}]
        builder = Inference._set_value(builder, values, 50)
        self.assertEqual(demes.Graph.fromdict(builder).pulses[0].time, 50)


class TestOptimize(TempFilesTestCase):
    """
    Fits a single parameter (the size of deme A) to the expected SFS under the
    true model, starting from a perturbed value.
    """

    def setUp(self):
        super().setUp()
        self.graph = self.write("graph.yaml", GRAPH)
        # start the optimization away from the true value of 2000
        self.start_graph = self.write(
            "start.yaml", GRAPH.replace("start_size: 2000", "start_size: 1500")
        )
        self.options = self.write(
            "options.yaml",
            "parameters:"
            + size_param("NA", "A")
            + "  lower_bound: 100\n  upper_bound: 100000\n",
        )
        self.fs = moments.Demes.SFS(
            demes.loads(GRAPH), samples={"A": 6, "B": 6}, theta=1000
        )

    def test_recovers_parameter(self):
        # Fit NA starting from 1500, to data from the true model (NA = 2000), with
        # each optimization method. Should land within 1% of the true value.
        for method in ["fmin", "powell", "lbfgsb"]:
            names, popt, LL = Inference.optimize(
                self.start_graph, self.options, self.fs, method=method
            )
            self.assertEqual(names, ["NA"])
            self.assertTrue(np.isclose(popt[0], 2000, rtol=1e-2), msg=method)

    def test_recovers_parameter_with_uL(self):
        # Same fit, but with a fixed uL (Poisson likelihood) instead of fitting theta.
        # theta = 4 * Ne * uL, with Ne the size of the root deme
        names, popt, LL = Inference.optimize(
            self.start_graph, self.options, self.fs, uL=1000 / 4 / 1000
        )
        self.assertTrue(np.isclose(popt[0], 2000, rtol=1e-2))

    def test_ancient_sample(self):
        # Same fit, with A sampled 50.5 generations ago. Ancient samples are given by
        # pop ids "{deme}_sampled_{gen}_{frac_gen}", here "A_sampled_50_5".
        fs = moments.Demes.SFS(
            demes.loads(GRAPH),
            sampled_demes=["A", "B"],
            sample_sizes=[6, 6],
            sample_times=[50.5, 0],
            theta=1000,
        )
        fs.pop_ids = ["A_sampled_50_5", "B"]
        names, popt, LL = Inference.optimize(self.start_graph, self.options, fs)
        self.assertTrue(np.isclose(popt[0], 2000, rtol=1e-2))

    def test_output(self):
        # The best-fit graph is written to `output`, and an existing output file
        # isn't overwritten (overwrite=False is the default).
        output = os.path.join(self.tmpdir.name, "best_fit.yaml")
        names, popt, LL = Inference.optimize(
            self.start_graph, self.options, self.fs, output=output
        )
        g = demes.load(output)
        self.assertEqual(g["A"].epochs[0].start_size, popt[0])
        # don't overwrite an existing file unless asked to
        with open(output, "w") as fout:
            fout.write("existing")
        with open(os.devnull, "w") as devnull:
            Inference.optimize(
                self.graph,
                self.options,
                self.fs,
                maxiter=1,
                output=output,
                output_stream=devnull,
            )
        with open(output) as fin:
            self.assertEqual(fin.read(), "existing")

    def test_invalid_inputs(self):
        # Each raises: data without pop_ids, unknown method, negative perturb, and
        # fitting ancestral misidentification with folded data.
        fs = self.fs.copy()
        fs.pop_ids = None
        with self.assertRaises(ValueError):
            Inference.optimize(self.graph, self.options, fs)
        with self.assertRaises(ValueError):
            Inference.optimize(self.graph, self.options, self.fs, method="newton")
        with self.assertRaises(ValueError):
            Inference.optimize(self.graph, self.options, self.fs, perturb=-1)
        with self.assertRaises(ValueError):
            Inference.optimize(
                self.graph, self.options, self.fs.fold(), fit_ancestral_misid=True
            )


class TestUncerts(TempFilesTestCase):
    def setUp(self):
        super().setUp()
        # uncerts() caches spectra between calls, keyed only by parameter values,
        # so clear it to keep tests independent of each other
        Inference._sfs_cache.clear()
        self.graph = self.write("graph.yaml", GRAPH)
        self.options = self.write("options.yaml", "parameters:" + size_param("NA", "A"))
        self.fs = moments.Demes.SFS(
            demes.loads(GRAPH), samples={"A": 6, "B": 6}, theta=1000
        )

    def test_fim(self):
        # FIM standard errors are finite and positive: two values without uL (NA and
        # theta, which is fit), and one value with uL.
        out = Inference.uncerts(self.graph, self.options, self.fs)
        self.assertEqual(len(out), 2)
        self.assertTrue(np.all(np.isfinite(out)) and np.all(out > 0))
        out = Inference.uncerts(self.graph, self.options, self.fs, uL=0.25)
        self.assertEqual(len(out), 1)
        self.assertTrue(np.isfinite(out[0]) and out[0] > 0)

    def test_gim(self):
        # GIM with 5 Poisson-resampled bootstrap spectra gives finite, positive
        # standard errors.
        rng = np.random.default_rng(1)
        bootstraps = [
            moments.Spectrum(rng.poisson(self.fs.data), pop_ids=self.fs.pop_ids)
            for _ in range(5)
        ]
        out = Inference.uncerts(
            self.graph, self.options, self.fs, bootstraps=bootstraps, method="GIM"
        )
        self.assertTrue(np.all(np.isfinite(out)) and np.all(out > 0))

    def test_output(self):
        # The uncertainties table written to `output` has the header, param name,
        # value, and the same standard error as returned.
        output = os.path.join(self.tmpdir.name, "uncerts.txt")
        out = Inference.uncerts(
            self.graph, self.options, self.fs, uL=0.25, output=output
        )
        with open(output) as fin:
            lines = fin.read().splitlines()
        self.assertEqual(lines[0], "#param\topt_value\tstd_err")
        name, value, std_err = lines[1].split("\t")
        self.assertEqual(name, "NA")
        self.assertEqual(float(value), 2000)
        self.assertTrue(np.isclose(float(std_err), out[0]))

    def test_invalid_inputs(self):
        # Each raises: GIM without bootstraps, unknown method, and fitting ancestral
        # misidentification without misid_fit.
        with self.assertRaises(ValueError):
            Inference.uncerts(self.graph, self.options, self.fs, method="GIM")
        with self.assertRaises(ValueError):
            Inference.uncerts(self.graph, self.options, self.fs, method="other")
        with self.assertRaises(ValueError):
            Inference.uncerts(
                self.graph, self.options, self.fs, fit_ancestral_misid=True
            )


class TestObjectiveFunctions(unittest.TestCase):
    """Behavior of the objective functions that the optimizers minimize."""

    def test_LD_nan_likelihood(self):
        # Mock the LD log-likelihood to return nan: the objective function returns
        # the out-of-bounds penalty (a large positive value) instead of nan.
        builder = demes.loads_asdict(GRAPH)
        options = {"parameters": [{"name": "NA", "values": SIZE_A}]}
        with unittest.mock.patch(
            "moments.LD.Inference.ll_over_bins", return_value=np.nan
        ):
            val = Inference._object_func_LD(
                np.array([2000.0]),
                None,  # means and varcovs aren't used with the mocked likelihood
                None,
                builder,
                options,
                pop_ids=["A", "B"],
                rs=[0, 1e-5],
                normalization="A",
            )
        self.assertEqual(val, -Inference._out_of_bounds_val)


if __name__ == "__main__":
    unittest.main()
