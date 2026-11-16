"""
Tests for moments.Demes.LRT.
"""

import os
import tempfile
import unittest
import warnings

import demes
import numpy as np

import moments
import moments.LD
from moments.Demes import LRT

# A split with migration, as two directed migrations that a model can fit with one
# shared rate. Sizes in individuals, times in generations.
GRAPH = """
time_units: generations
demes:
- name: anc
  epochs:
  - {{start_size: 10000, end_time: 2000}}
- name: A
  ancestors: [anc]
  epochs:
  - {{start_size: {NA0}, end_size: {NAF}}}
- name: B
  ancestors: [anc]
  epochs:
  - {{start_size: 20000}}
migrations:
- {{source: A, dest: B, rate: {m}}}
- {{source: B, dest: A, rate: {m}}}
"""

# options-file values for A's sizes, B's size, the split time and the migration rates
NA0 = "demes: {A: {epochs: {0: start_size}}}"
NAF = "demes: {A: {epochs: {0: end_size}}}"
NB = "demes: {B: {epochs: {0: start_size}}}"
T = "demes: {anc: {epochs: {0: end_time}}}"
M_AB = "migrations: {0: rate}"
M_BA = "migrations: {1: rate}"


def param(name, values, upper=None):
    # one parameter of an options file, as YAML lines
    lines = [f"- name: {name}", "  values:"] + [f"  - {v}" for v in values]
    if upper is not None:
        lines.append(f"  upper_bound: {upper}")
    return "\n".join(lines)


# the complex model fits one size for A, B's size, the split time and one
# migration rate; the nested model fixes the migration rate (at 0, its lower bound)
COMPLEX = [
    param("NA", [NA0, NAF]),
    param("NB", [NB]),
    param("T", [T]),
    param("m", [M_AB, M_BA], upper=1e-3),
]
NESTED = COMPLEX[:3]


class TempFiles(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def write(self, name, text):
        path = os.path.join(self.tmpdir.name, name)
        with open(path, "w") as fout:
            fout.write(text)
        return path

    def options(self, name, params):
        return self.write(name, "parameters:\n" + "\n".join(params) + "\n")

    def graph(self, name, NA0=2000, NAF=2000, m=0):
        return self.write(name, GRAPH.format(NA0=NA0, NAF=NAF, m=m))


class Nesting(TempFiles):
    def nesting(self, complex_params, nested_params):
        load = moments.Demes.Inference._get_params_dict
        return LRT._nesting(
            load(self.options("c.yaml", complex_params)),
            load(self.options("n.yaml", nested_params)),
        )

    def test_fixed_and_shared(self):
        # The complex model fits A's start and end sizes, B's size, the split time
        # and the two migration rates separately; the nested model shares one
        # parameter for A's sizes and one for the migration rates, and fixes nothing.
        complex_params = [
            param("NA0", [NA0]),
            param("NAF", [NAF]),
            param("NB", [NB]),
            param("T", [T]),
            param("m_AB", [M_AB]),
            param("m_BA", [M_BA]),
        ]
        nested_params = [param("NA", [NA0, NAF])] + complex_params[2:4]
        nested_params += [param("m", [M_AB, M_BA])]
        nest = self.nesting(complex_params, nested_params)
        self.assertEqual(nest["fixed"], [])
        self.assertEqual(sorted(nest["shared"]), [[0, 1], [4, 5]])

    def test_fixed(self):
        # A complex parameter whose values no nested parameter sets is fixed.
        nest = self.nesting(COMPLEX, NESTED)
        self.assertEqual((nest["fixed"], nest["shared"]), ([3], []))

    def test_not_nested(self):
        # Errors: the nested model splits a parameter the complex model shares, fits
        # a value the complex model doesn't, or fixes and shares nothing.
        complex_params = [param("NA", [NA0, NAF])]
        for nested_params in [
            [param("NA0", [NA0]), param("NAF", [NAF])],
            [param("NA", [NA0, NAF]), param("NB", [NB])],
            complex_params,
        ]:
            with self.assertRaises(ValueError):
                self.nesting(complex_params, nested_params)


class SFSWellSpecified(TempFiles):
    """
    The data are the expected SFS without migration (the nested model), with Poisson
    bootstraps, so the composite likelihood is the true likelihood and the adjustment
    is close to 1. Migration is fixed at its lower bound, 0: central differences
    would step to negative rates, which give an invalid graph, so these tests also
    check that its derivatives are one-sided.
    """

    def setUp(self):
        super().setUp()
        self.graph_file = self.graph("g.yaml")
        self.complex = self.options("c.yaml", COMPLEX)
        self.nested = self.options("n.yaml", NESTED)
        self.data = moments.Demes.SFS(
            demes.load(self.graph_file), ["A", "B"], [16, 16], theta=4000
        )
        rng = np.random.default_rng(1)
        self.boots = [
            moments.Spectrum(rng.poisson(self.data.data), mask=self.data.mask)
            for _ in range(200)
        ]

    def adjust(self, data=None, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LRT.LRT_adjust(
                self.graph_file,
                self.complex,
                self.graph_file,
                self.nested,
                self.data if data is None else data,
                self.boots,
                **kwargs,
            )

    def test_adjustment_near_one(self):
        # With theta fit by multinomial scaling, and with the true uL (theta = 4 N_anc
        # uL), the adjustment is close to 1.
        for uL in [None, 4000 / (4 * 10000)]:
            adj = self.adjust(uL=uL)
            self.assertTrue(abs(adj - 1) < 0.2, msg=f"uL {uL}: adj = {adj}")

    def test_return_matrices(self):
        # With return_matrices, H, J and the mean score are over all parameters (the
        # four demographic ones and theta), and the nested parameter is m (index 3).
        adj, H, J, cU, nested_indices = self.adjust(return_matrices=True)
        self.assertEqual(H.shape, (5, 5))
        self.assertEqual(J.shape, (5, 5))
        self.assertEqual(cU.shape, (5, 1))
        self.assertEqual(nested_indices, [3])
        sub = np.ix_(nested_indices, nested_indices)
        self.assertTrue(np.isclose(adj, 1 / np.trace(J[sub] @ np.linalg.inv(H[sub]))))

    def test_ancestral_misid(self):
        # Data with ancestral states misidentified at 2%, tested with the
        # misidentification probability as a free parameter: the adjustment is
        # still close to 1. Without misid_fit, it raises.
        data = moments.Misc.flip_ancestral_misid(self.data, 0.02)
        adj = self.adjust(data=data, fit_ancestral_misid=True, misid_fit=0.02)
        self.assertTrue(abs(adj - 1) < 0.2, msg=f"adj = {adj}")
        with self.assertRaises(ValueError):
            self.adjust(fit_ancestral_misid=True)


class SFSShared(TempFiles):
    def test_constant_size(self):
        # The nested model has one size for A's start and end (constant size) within
        # a complex model with a size change: the nested parameter is the log ratio
        # of the end to the start size, and the adjustment is close to 1, as the
        # data are the expected SFS of the nested model.
        graph = self.graph("g.yaml", m=1e-5)
        complex_params = [param("NA0", [NA0]), param("NAF", [NAF])] + COMPLEX[1:]
        complex_file = self.options("c.yaml", complex_params)
        nested_file = self.options("n.yaml", COMPLEX)
        data = moments.Demes.SFS(demes.load(graph), ["A", "B"], [16, 16], theta=4000)
        rng = np.random.default_rng(2)
        boots = [moments.Spectrum(rng.poisson(data.data)) for _ in range(200)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj, H, J, cU, nested_indices = LRT.LRT_adjust(
                graph,
                complex_file,
                graph,
                nested_file,
                data,
                boots,
                return_matrices=True,
            )
        self.assertEqual(nested_indices, [1])
        self.assertTrue(abs(adj - 1) < 0.2, msg=f"adj = {adj}")


class LDWellSpecified(TempFiles):
    def test_no_migration_true(self):
        # As moments' GodambeWellSpecified test, through Demes files: the data are the
        # expected statistics without migration, and bootstraps shift one statistic
        # at a time by +/- sqrt(K) SDs, so their variances equal varcovs and the
        # composite likelihood is the true one: the adjustment is about 1 (0.95 at
        # the default eps = 0.01, and 1.000 at eps = 1e-3).
        graph = self.graph("g.yaml")
        rs = [0, 1e-6, 1e-5, 1e-4]
        y = moments.Demes.Inference.compute_bin_stats(
            demes.load(graph), ["A", "B"], rs=rs
        )
        means = [np.array(m) for m in moments.LD.Inference.sigmaD2(y)]
        names = moments.LD.Util.moment_names(2)
        norm = [names[0].index("pi2_0_0_0_0"), names[1].index("H_0_0")]
        sds = []
        for i, m in enumerate(means):
            sd = 0.02 * np.abs(m)
            sd[norm[0] if i < len(means) - 1 else norm[1]] = 0
            sds.append(sd)
        varcovs = [np.diag(sd**2) for sd in sds]
        coords = [(i, j) for i, sd in enumerate(sds) for j in range(len(sd)) if sd[j]]
        boots = []
        for i, j in coords:
            for sign in (1, -1):
                boot = [m.copy() for m in means]
                boot[i][j] += sign * np.sqrt(len(coords)) * sds[i][j]
                boots.append(boot)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adj = LRT.LRT_adjust_LD(
                graph,
                self.options("c.yaml", COMPLEX),
                graph,
                self.options("n.yaml", NESTED),
                means,
                varcovs,
                boots,
                ["A", "B"],
                rs,
                "A",
            )
        self.assertTrue(np.isclose(adj, 1, rtol=0.1), msg=f"adj = {adj}")


if __name__ == "__main__":
    unittest.main()
