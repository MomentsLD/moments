import os
import unittest
import math
import pathlib

import numpy as np
import moments
from moments.Demes import Demes
import time

import demes
import warnings


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_two_way_split(self):
        ns0, ns1 = 10, 6
        pop_ids = ["A", "B"]
        fs = moments.Demographics1D.snm([ns0 + ns1], pop_ids=["O"])
        out = Demes._split_fs(fs, 0, pop_ids, [ns0, ns1])
        fs = fs.split(0, ns0, ns1, new_ids=pop_ids)
        self.assertTrue(np.all([x == y for x, y in zip(pop_ids, out.pop_ids)]))
        self.assertTrue(np.all(out.sample_sizes == np.array([ns0, ns1])))
        self.assertTrue(np.allclose(out.data, fs.data))

        rho = [0, 1]
        y = moments.LD.Demographics1D.snm(rho=rho, pop_ids=["O"])
        out = Demes._apply_LD_event(y, ("split", "O", pop_ids), 0, pop_ids)
        y = y.split(0, new_ids=pop_ids)
        self.assertTrue(np.all([x == y for x, y in zip(pop_ids, out.pop_ids)]))
        for s0, s1 in zip(y, out):
            self.assertTrue(np.allclose(s0, s1))

    def test_three_way_split_1(self):
        pop_ids = ["A", "B"]
        child_ids = ["C1", "C2", "C3"]
        child_sizes = [4, 6, 8]
        nsA = 5
        nsB = sum(child_sizes)
        fs = moments.Spectrum(np.ones((nsA + 1, nsB + 1)), pop_ids=pop_ids)
        out = Demes._split_fs(fs, 1, child_ids, child_sizes)
        self.assertTrue(out.Npop == 4)
        self.assertTrue(
            np.all([x == y for x, y in zip(out.sample_sizes, [nsA] + child_sizes)])
        )
        self.assertTrue(
            np.all([x == y for x, y in zip(out.pop_ids, ["A"] + child_ids)])
        )

        rho = [0, 1, 10]
        y = moments.LD.LDstats(
            [np.random.rand(15) for _ in range(len(rho))] + [np.random.rand(3)],
            pop_ids=["A", "B"],
            num_pops=2,
        )
        out = Demes._apply_LD_event(y, ("split", "B", child_ids), 0, ["A"] + child_ids)
        self.assertTrue(out.num_pops == 4)
        self.assertTrue(
            np.all([i == j for i, j in zip(out.pop_ids, ["A"] + child_ids)])
        )

    def test_three_way_split_0(self):
        pop_ids = ["A", "B"]
        child_ids = ["C1", "C2", "C3"]
        child_sizes = [4, 6, 8]
        nsA = sum(child_sizes)
        nsB = 5
        fs = moments.Spectrum(np.ones((nsA + 1, nsB + 1)), pop_ids=pop_ids)
        out = Demes._split_fs(fs, 0, child_ids, child_sizes)
        self.assertTrue(out.Npop == 4)
        self.assertTrue(
            np.all([x == y for x, y in zip(out.sample_sizes, [4, 5, 6, 8])])
        )
        self.assertTrue(
            np.all([x == y for x, y in zip(out.pop_ids, ["C1", "B", "C2", "C3"])])
        )

        rho = [0, 1, 10]
        y = moments.LD.LDstats(
            [np.random.rand(15) for _ in range(len(rho))] + [np.random.rand(3)],
            pop_ids=["A", "B"],
            num_pops=2,
        )
        out = Demes._apply_LD_event(y, ("split", "A", child_ids), 0, ["B"] + child_ids)
        self.assertTrue(out.num_pops == 4)
        self.assertTrue(
            np.all([i == j for i, j in zip(out.pop_ids, ["C1", "B", "C2", "C3"])])
        )


class TestReorder(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_reorder_three_pops(self):
        fs = moments.Spectrum(
            np.random.rand(4 * 6 * 8).reshape((4, 6, 8)), pop_ids=["A", "B", "C"]
        )
        y = moments.LD.LDstats(
            [np.random.rand(45)] + [np.random.rand(6)],
            num_pops=3,
            pop_ids=["A", "B", "C"],
        )
        new_orders = [[1, 0, 2], [0, 2, 1], [2, 1, 0], [2, 0, 1]]
        for new_order in new_orders:
            new_ids = [fs.pop_ids[ii] for ii in new_order]
            out = Demes._reorder_fs(fs, new_ids)
            out_ld = Demes._reorder_LD(y, new_ids)
            self.assertTrue(
                np.all(
                    [
                        x == y
                        for x, y in zip((fs.shape[ii] for ii in new_order), out.shape)
                    ]
                )
            )
            self.assertTrue(
                np.all(
                    [
                        x == y
                        for x, y in zip(
                            (fs.pop_ids[ii] for ii in new_order), out.pop_ids
                        )
                    ]
                )
            )
            self.assertTrue(
                np.all(
                    [
                        x == y
                        for x, y in zip(
                            (y.pop_ids[ii] for ii in new_order), out_ld.pop_ids
                        )
                    ]
                )
            )

    def test_reorder_five_pops(self):
        fs = moments.Spectrum(
            np.random.rand(4 * 5 * 6 * 7 * 8).reshape((4, 5, 6, 7, 8)),
            pop_ids=["A", "B", "C", "D", "E"],
        )
        y = moments.LD.LDstats(
            [np.random.rand(210)] + [np.random.rand(15)],
            num_pops=5,
            pop_ids=["A", "B", "C", "D", "E"],
        )
        for new_order_idx in range(10):
            new_order = np.random.permutation([0, 1, 2, 3, 4])
            new_ids = [fs.pop_ids[ii] for ii in new_order]
            out = Demes._reorder_fs(fs, new_ids)
            out_ld = Demes._reorder_LD(y, new_ids)
            self.assertTrue(
                np.all(
                    [
                        x == y
                        for x, y in zip((fs.shape[ii] for ii in new_order), out.shape)
                    ]
                )
            )
            self.assertTrue(
                np.all(
                    [
                        x == y
                        for x, y in zip(
                            (fs.pop_ids[ii] for ii in new_order), out.pop_ids
                        )
                    ]
                )
            )
            self.assertTrue(
                np.all(
                    [
                        x == y
                        for x, y in zip(
                            (y.pop_ids[ii] for ii in new_order), out_ld.pop_ids
                        )
                    ]
                )
            )


class TestAdmix(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_two_way_admixture(self):
        fs = moments.Spectrum(np.ones((7, 7, 7)), pop_ids=["A", "B", "C"])
        parents = ["A", "C"]
        proportions = [0.3, 0.7]
        child = "D"
        child_size = 2
        out = Demes._admix_fs(fs, parents, proportions, child, child_size)
        self.assertTrue(
            np.all([x == y for x, y in zip(out.sample_sizes, (4, 6, 4, 2))])
        )
        self.assertTrue(
            np.all([x == y for x, y in zip(out.pop_ids, ("A", "B", "C", "D"))])
        )

        child_size = 6
        out = Demes._admix_fs(fs, parents, proportions, child, child_size)
        self.assertTrue(np.all([x == y for x, y in zip(out.sample_sizes, (6, 6))]))
        self.assertTrue(np.all([x == y for x, y in zip(out.pop_ids, ("B", "D"))]))

        y = moments.LD.LDstats(
            [np.random.rand(45)] + [np.random.rand(6)],
            num_pops=3,
            pop_ids=["A", "B", "C"],
        )
        out = Demes._admix_LD(y, parents, proportions, child, marginalize=False)
        self.assertTrue(
            np.all([x == y for x, y in zip(out.pop_ids, ("A", "B", "C", "D"))])
        )
        out = Demes._admix_LD(y, parents, proportions, child, marginalize=True)
        self.assertTrue(np.all([x == y for x, y in zip(out.pop_ids, ("B", "D"))]))

    def test_three_way_admixture(self):
        fs = moments.Spectrum(np.ones((5, 5, 5)), pop_ids=["A", "B", "C"])
        parents = ["A", "B", "C"]
        proportions = [0.2, 0.3, 0.5]
        child = "D"
        child_size = 2
        out = Demes._admix_fs(fs, parents, proportions, child, child_size)
        self.assertTrue(
            np.all([x == y for x, y in zip(out.sample_sizes, (2, 2, 2, 2))])
        )
        self.assertTrue(
            np.all([x == y for x, y in zip(out.pop_ids, ("A", "B", "C", "D"))])
        )

        child_size = 4
        out = Demes._admix_fs(fs, parents, proportions, child, child_size)
        self.assertTrue(np.all([x == y for x, y in zip(out.sample_sizes, (4,))]))
        self.assertTrue(np.all([x == y for x, y in zip(out.pop_ids, ("D",))]))

        y = moments.LD.LDstats(
            [np.random.rand(45)] + [np.random.rand(6)],
            num_pops=3,
            pop_ids=["A", "B", "C"],
        )
        out = Demes._admix_LD(y, parents, proportions, child, marginalize=False)
        self.assertTrue(
            np.all([x == y for x, y in zip(out.pop_ids, ("A", "B", "C", "D"))])
        )
        out = Demes._admix_LD(y, parents, proportions, child, marginalize=True)
        self.assertTrue(np.all([x == y for x, y in zip(out.pop_ids, ("D",))]))


class TestPulse(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_pulse_function(self):
        n0 = 30
        n1 = 10
        fs = moments.Demographics2D.snm([n0, n1], pop_ids=["A", "B"])
        out = Demes._pulse_fs(fs, ["A"], "B", [0.2])
        self.assertTrue(out.sample_sizes[0] == n0 - n1)
        self.assertTrue(out.sample_sizes[1] == n1)
        fs2 = fs.pulse_migrate(0, 1, n0 - n1, 0.2)
        self.assertTrue(np.allclose(fs2.data, out.data))


class CompareOOA(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_direct_comparison(self):
        Ne = 7300
        gens = 25
        nuA = 12300 / Ne
        TA = (220e3 - 140e3) / 2 / Ne / gens
        nuB = 2100 / Ne
        TB = (140e3 - 21.2e3) / 2 / Ne / gens
        nuEu0 = 1000 / Ne
        nuEuF = 29725 / Ne
        nuAs0 = 510 / Ne
        nuAsF = 54090 / Ne
        TF = 21.2e3 / 2 / Ne / gens
        mAfB = 2 * Ne * 25e-5
        mAfEu = 2 * Ne * 3e-5
        mAfAs = 2 * Ne * 1.9e-5
        mEuAs = 2 * Ne * 9.6e-5

        ns = [10, 10, 10]

        fs_moments = moments.Demographics3D.out_of_Africa(
            (
                nuA,
                TA,
                nuB,
                TB,
                nuEu0,
                nuEuF,
                nuAs0,
                nuAsF,
                TF,
                mAfB,
                mAfEu,
                mAfAs,
                mEuAs,
            ),
            ns,
        )
        fs_demes = moments.Spectrum.from_demes(
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yaml"),
            ["YRI", "CEU", "CHB"],
            ns,
        )

        self.assertTrue(
            np.all([x == y for x, y in zip(fs_moments.pop_ids, fs_demes.pop_ids)])
        )
        self.assertTrue(np.allclose(fs_demes.data, fs_moments.data))

    def test_direct_comparison_LD(self):
        Ne = 7300
        gens = 25
        nuA = 12300 / Ne
        TA = (220e3 - 140e3) / 2 / Ne / gens
        nuB = 2100 / Ne
        TB = (140e3 - 21.2e3) / 2 / Ne / gens
        nuEu0 = 1000 / Ne
        nuEuF = 29725 / Ne
        nuAs0 = 510 / Ne
        nuAsF = 54090 / Ne
        TF = 21.2e3 / 2 / Ne / gens
        mAfB = 2 * Ne * 25e-5
        mAfEu = 2 * Ne * 3e-5
        mAfAs = 2 * Ne * 1.9e-5
        mEuAs = 2 * Ne * 9.6e-5

        rho = [0, 1, 10]
        theta = 0.001

        y_moments = moments.LD.Demographics3D.out_of_Africa(
            (
                nuA,
                TA,
                nuB,
                TB,
                nuEu0,
                nuEuF,
                nuAs0,
                nuAsF,
                TF,
                mAfB,
                mAfEu,
                mAfAs,
                mEuAs,
            ),
            rho=rho,
            theta=theta,
        )
        y_demes = moments.LD.LDstats.from_demes(
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yaml"),
            ["YRI", "CEU", "CHB"],
            rho=rho,
            theta=theta,
        )

        self.assertTrue(
            np.all([x == y for x, y in zip(y_moments.pop_ids, y_demes.pop_ids)])
        )
        for x, y in zip(y_demes, y_moments):
            self.assertTrue(np.allclose(x, y))


class ComputeFromGraphs(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_admixture_with_marginalization(self):
        # admixture scenario where two populations admix, one continues and the other is marginalized
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=100, end_time=100)])
        b.add_deme(
            "A",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "B",
            epochs=[dict(start_size=100, end_time=50)],
            ancestors=["anc"],
        )
        b.add_deme(
            "C",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["A", "B"],
            proportions=[0.75, 0.25],
            start_time=50,
        )
        g = b.resolve()

        fs = moments.Demographics1D.snm([30])
        fs = fs.split(0, 20, 10)
        fs.integrate([1, 1], 50 / 2 / 100)
        fs = fs.admix(0, 1, 10, 0.75)
        fs.integrate([1, 1], 50 / 2 / 100)
        fs_demes = Demes.SFS(
            g, sampled_demes=["A", "C"], sample_sizes=[10, 10], theta=1
        )
        self.assertTrue(np.allclose(fs.data, fs_demes.data))

        y = moments.LD.Demographics1D.snm()
        y = y.split(0)
        y.integrate([1, 1], 50 / 2 / 100)
        y = y.admix(0, 1, 0.75)
        y = y.marginalize([1])
        y.integrate([1, 1], 50 / 2 / 100)
        y_demes = Demes.LD(g, sampled_demes=["A", "C"])

    def test_migration_end_and_marginalize(self):
        # ghost population with migration that ends before time zero
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=100, end_time=100)])
        b.add_deme(
            "A",
            epochs=[dict(start_size=100, end_time=25)],
            ancestors=["anc"],
        )
        b.add_deme(
            "B",
            epochs=[dict(start_size=100, end_time=50)],
            ancestors=["anc"],
        )
        b.add_deme(
            "C",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["B"],
        )
        b.add_deme(
            "D",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["B"],
        )
        b.add_migration
        g = b.resolve()

        fs_demes = Demes.SFS(g, sampled_demes=["C", "D"], sample_sizes=[4, 4], theta=1)
        y_demes = Demes.LD(g, sampled_demes=["C", "D"])
        self.assertTrue(fs_demes.ndim == 2)
        self.assertTrue(np.all([x == y for x, y in zip(fs_demes.pop_ids, ["C", "D"])]))
        self.assertTrue(y_demes.num_pops == 2)
        self.assertTrue(np.all([x == y for x, y in zip(y_demes.pop_ids, ["C", "D"])]))


def moments_ooa(ns):
    fs = moments.Demographics1D.snm([sum(ns)])
    fs.integrate([1.6849315068493151], 0.2191780821917808)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1] + ns[2])
    fs.integrate(
        [1.6849315068493151, 0.2876712328767123],
        0.3254794520547945,
        m=[[0, 3.65], [3.65, 0]],
    )
    fs = moments.Manips.split_2D_to_3D_2(fs, ns[1], ns[2])

    def nu_func(t):
        return [
            1.6849315068493151,
            0.136986301369863
            * np.exp(
                np.log(4.071917808219178 / 0.136986301369863) * t / 0.05808219178082192
            ),
            0.06986301369863014
            * np.exp(
                np.log(7.409589041095891 / 0.06986301369863014)
                * t
                / 0.05808219178082192
            ),
        ]

    fs.integrate(
        nu_func,
        0.05808219178082192,
        m=[[0, 0.438, 0.2774], [0.438, 0, 1.4016], [0.2774, 1.4016, 0]],
    )
    return fs


class TestMomentsSFS(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_num_lineages(self):
        # simple merge model
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=100, end_time=100)])
        b.add_deme(
            "pop1",
            epochs=[dict(start_size=100, end_time=10)],
            ancestors=["anc"],
        )
        b.add_deme(
            "pop2",
            epochs=[dict(start_size=100, end_time=10)],
            ancestors=["anc"],
        )
        b.add_deme(
            "pop3",
            epochs=[dict(start_size=100, end_time=10)],
            ancestors=["anc"],
        )
        b.add_deme(
            "pop",
            ancestors=["pop1", "pop2", "pop3"],
            proportions=[0.1, 0.2, 0.7],
            start_time=10,
            epochs=[dict(end_time=0, start_size=100)],
        )
        g = b.resolve()
        sampled_demes = ["pop"]
        demes_demo_events = g.discrete_demographic_events()
        demo_events, demes_present = Demes._get_demographic_events(
            g, demes_demo_events, sampled_demes
        )
        deme_sample_sizes = Demes._get_deme_sample_sizes(
            g, demo_events, sampled_demes, [20], demes_present
        )
        self.assertTrue(deme_sample_sizes[(math.inf, 100)][0] == 60)
        self.assertTrue(
            np.all([deme_sample_sizes[(100, 10)][i] == 20 for i in range(3)])
        )

        # simple admix model
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=100, end_time=100)])
        b.add_deme(
            "pop1",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "pop2",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "pop3",
            epochs=[dict(start_size=100, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "pop",
            ancestors=["pop1", "pop2", "pop3"],
            proportions=[0.1, 0.2, 0.7],
            start_time=10,
            epochs=[dict(start_size=100, end_time=0)],
        )
        g = b.resolve()
        sampled_demes = ["pop"]
        demes_demo_events = g.discrete_demographic_events()
        demo_events, demes_present = Demes._get_demographic_events(
            g, demes_demo_events, sampled_demes
        )
        deme_sample_sizes = Demes._get_deme_sample_sizes(
            g, demo_events, sampled_demes, [20], demes_present, unsampled_n=10
        )
        self.assertTrue(deme_sample_sizes[(math.inf, 100)][0] == 90)
        self.assertTrue(
            np.all([deme_sample_sizes[(100, 10)][i] == 30 for i in range(3)])
        )

    # test basic results against moments implementation
    def test_one_pop(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Pop", epochs=[dict(start_size=1000, end_time=0)])
        g = b.resolve()
        fs = Demes.SFS(g, ["Pop"], [20], theta=1)
        fs_m = moments.Demographics1D.snm([20])
        self.assertTrue(np.allclose(fs.data, fs_m.data))

        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Pop",
            epochs=[
                dict(start_size=1000, end_time=2000),
                dict(end_time=0, start_size=10000),
            ],
        )
        g = b.resolve()
        fs = Demes.SFS(g, ["Pop"], [20], theta=1)
        fs_m = moments.Demographics1D.snm([20])
        fs_m.integrate([10], 1)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_more_than_5_demes(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=1000)])
        for i in range(6):
            b.add_deme(
                f"pop{i}",
                epochs=[dict(start_size=1000, end_time=0)],
                ancestors=["anc"],
            )
        g = b.resolve()
        with self.assertRaises(ValueError):
            Demes.SFS(g, ["pop{i}" for i in range(6)], [10 for i in range(6)], theta=1)

        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=1000)])
        for i in range(3):
            b.add_deme(
                f"pop{i}",
                epochs=[dict(start_size=1000, end_time=0)],
                ancestors=["anc"],
            )
        g = b.resolve()
        with self.assertRaises(ValueError):
            Demes.SFS(
                g,
                ["pop{i}" for i in range(3)],
                [10 for i in range(3)],
                sample_times=[5, 10, 15],
                theta=1,
            )

    def test_one_pop_ancient_samples(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Pop", epochs=[dict(start_size=1000, end_time=0)])
        g = b.resolve()
        fs = Demes.SFS(g, ["Pop", "Pop"], [20, 4], sample_times=[0, 100], theta=1)
        fs_m = moments.Demographics1D.snm([24])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 4)
        fs_m.integrate([1, 1], 100 / 2 / 1000, frozen=[False, True])
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_merge(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Anc", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme(
            "Source1",
            epochs=[dict(start_size=2000, end_time=10)],
            ancestors=["Anc"],
        )
        b.add_deme(
            "Source2",
            epochs=[dict(start_size=3000, end_time=10)],
            ancestors=["Anc"],
        )
        b.add_deme(
            "Pop",
            ancestors=["Source1", "Source2"],
            proportions=[0.8, 0.2],
            start_time=10,
            epochs=[dict(start_size=4000, end_time=0)],
        )
        g = b.resolve()
        fs = Demes.SFS(g, ["Pop"], [20], theta=1)

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m.integrate([2, 3], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 20, 0.8)
        fs_m.integrate([4], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_admixture(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Anc", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme(
            "Source1",
            epochs=[dict(start_size=2000, end_time=0)],
            ancestors=["Anc"],
        )
        b.add_deme(
            "Source2",
            epochs=[dict(start_size=3000, end_time=0)],
            ancestors=["Anc"],
        )
        b.add_deme(
            "Pop",
            ancestors=["Source1", "Source2"],
            proportions=[0.8, 0.2],
            start_time=10,
            epochs=[dict(start_size=4000, end_time=0)],
        )
        g = b.resolve()
        fs = Demes.SFS(g, ["Source1", "Source2", "Pop"], [10, 10, 10], theta=1)

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m.integrate([2, 3], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.8)
        fs_m.integrate([2, 3, 4], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_growth_models(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Pop",
            epochs=[
                dict(end_time=1000, start_size=1000),
                dict(start_size=500, end_size=5000, end_time=0),
            ],
        )
        g = b.resolve()
        fs = Demes.SFS(g, ["Pop"], [100], theta=1)

        fs_m = moments.Demographics1D.snm([100])

        def nu_func(t):
            return [0.5 * np.exp(np.log(5000 / 500) * t / 0.5)]

        fs_m.integrate(nu_func, 0.5)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

        # Linear size functions are not currently supported in Demes
        # b = demes.Builder(description="test", time_units="generations")
        # b.add_deme(
        #    "Pop",
        #    epochs=[
        #        dict(end_time=1000, start_size=1000),
        #        dict(
        #            start_size=500, end_size=5000, end_time=0, size_function="linear",
        #        ),
        #    ],
        # )
        # g = b.resolve()
        # fs = Demes.SFS(g, ["Pop"], [100])
        #
        # fs_m = moments.Demographics1D.snm([100])
        #
        # def nu_func(t):
        #    return [0.5 + t / 0.5 * (5 - 0.5)]
        #
        # fs_m.integrate(nu_func, 0.5)
        # self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_pulse_model(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme(
            "source",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "dest",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_pulse(sources=["source"], dest="dest", time=10, proportions=[0.1])
        g = b.resolve()
        fs = Demes.SFS(g, ["source", "dest"], [20, 20], theta=1)

        fs_m = moments.Demographics1D.snm([60])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 40, 20)
        fs_m.integrate([1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_inplace(fs_m, 0, 1, 20, 0.1)
        fs_m.integrate([1, 1], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_n_way_split(self):
        b = demes.Builder(description="three-way", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=10)])
        b.add_deme(
            "deme1",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "deme2",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "deme3",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        g = b.resolve()
        ns = [10, 15, 20]
        fs = Demes.SFS(g, ["deme1", "deme2", "deme3"], ns, theta=1)
        self.assertTrue(np.all([fs.sample_sizes[i] == ns[i] for i in range(len(ns))]))

        fs_m1 = moments.Demographics1D.snm([sum(ns)])
        fs_m1 = moments.Manips.split_1D_to_2D(fs_m1, ns[0], ns[1] + ns[2])
        fs_m1 = moments.Manips.split_2D_to_3D_2(fs_m1, ns[1], ns[2])
        fs_m1.integrate([1, 1, 1], 10 / 2 / 1000)

        fs_m2 = moments.Demographics1D.snm([sum(ns)])
        fs_m2 = moments.Manips.split_1D_to_2D(fs_m2, ns[0] + ns[1], ns[2])
        fs_m2 = moments.Manips.split_2D_to_3D_1(fs_m2, ns[0], ns[1])
        fs_m2 = fs_m2.swapaxes(1, 2)
        fs_m2.integrate([1, 1, 1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs.data, fs_m1.data))
        self.assertTrue(np.allclose(fs.data, fs_m2.data))

    def test_n_way_admixture(self):
        b = demes.Builder(description="three-way merge", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme(
            "source1",
            epochs=[dict(start_size=1000, end_time=10)],
            ancestors=["anc"],
        )
        b.add_deme(
            "source2",
            epochs=[dict(start_size=1000, end_time=10)],
            ancestors=["anc"],
        )
        b.add_deme(
            "source3",
            epochs=[dict(start_size=1000, end_time=10)],
            ancestors=["anc"],
        )
        b.add_deme(
            "merged",
            ancestors=["source1", "source2", "source3"],
            proportions=[0.5, 0.2, 0.3],
            start_time=10,
            epochs=[dict(start_size=1000, end_time=0)],
        )
        g = b.resolve()
        ns = [10]
        fs = Demes.SFS(g, ["merged"], ns, theta=1)

        fs_m = moments.Demographics1D.snm([30])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 10, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.3)
        fs_m.integrate([1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs_m.data, fs.data))

        b = demes.Builder(description="three-way admix", time_units="generations")
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme(
            "source1",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "source2",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "source3",
            epochs=[dict(start_size=1000, end_time=0)],
            ancestors=["anc"],
        )
        b.add_deme(
            "admixed",
            ancestors=["source1", "source2", "source3"],
            proportions=[0.5, 0.2, 0.3],
            start_time=10,
            epochs=[dict(start_size=1000, end_time=0)],
        )
        g = b.resolve()
        ns = [10]
        fs = Demes.SFS(g, ["admixed"], ns, theta=1)

        fs_m = moments.Demographics1D.snm([30])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 10, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.3)
        fs_m.integrate([1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs_m.data[1:-1], fs.data[1:-1]))

        fs = Demes.SFS(g, ["source1", "admixed"], [10, 10], theta=1)

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 1, 2, 10, 0.3)
        fs_m.integrate([1, 1], 10 / 2 / 1000)

        fs[0, 0] = fs[-1, -1] = 0
        fs_m[0, 0] = fs_m[-1, -1] = 0
        self.assertTrue(np.allclose(fs_m.data[1:-1], fs.data[1:-1]))


class TestConcurrentEvents(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_branches_at_same_time(self):
        def from_old_style(sample_sizes):
            fs = moments.Demographics1D.snm([4 + sum(sample_sizes)])
            fs = fs.branch(0, sample_sizes[0])
            fs = fs.branch(0, sample_sizes[1])
            fs.integrate([1, 1, 1], 0.5)
            fs = fs.marginalize([0])
            return fs

        b = demes.Builder()
        b.add_deme("x", epochs=[dict(start_size=100)])
        b.add_deme("a", ancestors=["x"], start_time=100, epochs=[dict(start_size=100)])
        b.add_deme("b", ancestors=["x"], start_time=100, epochs=[dict(start_size=100)])
        graph = b.resolve()

        ns = [10, 10]
        fs_demes = moments.Spectrum.from_demes(
            graph, sampled_demes=["a", "b"], sample_sizes=ns
        )

        fs_moments = from_old_style(ns)

        self.assertTrue(np.allclose(fs_demes.data, fs_moments.data))

        b2 = demes.Builder()
        b2.add_deme("x", epochs=[dict(start_size=100)])
        b2.add_deme("a", ancestors=["x"], start_time=100, epochs=[dict(start_size=100)])
        b2.add_deme(
            "b", ancestors=["x"], start_time=99.9999, epochs=[dict(start_size=100)]
        )
        graph2 = b2.resolve()

        fs_demes2 = moments.Spectrum.from_demes(
            graph2, sampled_demes=["a", "b"], sample_sizes=ns
        )

        self.assertTrue(
            np.all([a == b for a, b, in zip(fs_demes.pop_ids, fs_demes2.pop_ids)])
        )
        self.assertTrue(np.allclose(fs_demes.data, fs_demes2.data))

    def test_concurrent_pulses_AtoB_BtoC(self):
        b = demes.Builder()
        b.add_deme("x", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme("a", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("b", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("c", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_pulse(sources=["a"], dest="b", time=50, proportions=[0.2])
        b.add_pulse(sources=["b"], dest="c", time=50, proportions=[0.2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = b.resolve()

        n = 20
        fs = moments.Spectrum.from_demes(graph, sampled_demes=["c"], sample_sizes=[n])

        self.assertEqual(fs.Npop, 1)
        self.assertEqual(fs.sample_sizes[0], n)
        self.assertEqual(fs.pop_ids[0], "c")

    def test_concurrent_pulses_AtoB_AtoC(self):
        b = demes.Builder()
        b.add_deme("x", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme("a", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("b", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("c", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_pulse(sources=["a"], dest="b", time=50, proportions=[0.2])
        b.add_pulse(sources=["a"], dest="c", time=50, proportions=[0.2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = b.resolve()

        n = 20
        fs = moments.Spectrum.from_demes(
            graph, sampled_demes=["b", "c"], sample_sizes=[n, n]
        )

        self.assertEqual(fs.Npop, 2)
        self.assertEqual(fs.sample_sizes[0], n)
        self.assertEqual(fs.sample_sizes[1], n)
        self.assertEqual(fs.pop_ids[0], "b")
        self.assertEqual(fs.pop_ids[1], "c")

    def test_concurrent_pulses_AtoC_BtoC(self):
        b = demes.Builder()
        b.add_deme("x", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme("a", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("b", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("c", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_pulse(sources=["a"], dest="c", time=50, proportions=[0.2])
        b.add_pulse(sources=["b"], dest="c", time=50, proportions=[0.2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = b.resolve()

        n = 20
        fs = moments.Spectrum.from_demes(graph, sampled_demes=["c"], sample_sizes=[n])

        self.assertEqual(fs.Npop, 1)
        self.assertEqual(fs.sample_sizes[0], n)
        self.assertEqual(fs.pop_ids[0], "c")

    def test_multipulse(self):
        b = demes.Builder()
        b.add_deme("x", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme("a", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("b", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_deme("c", ancestors=["x"], epochs=[dict(start_size=1000)])
        b.add_pulse(sources=["a", "b"], dest="c", time=50, proportions=[0.1, 0.2])
        graph = b.resolve()

        n = [8, 8, 8]
        fs = moments.Spectrum.from_demes(
            graph, sampled_demes=["a", "b", "c"], sample_sizes=n
        )

        fs2 = moments.Demographics1D.snm([16 + 16 + 8])
        fs2 = fs2.split(0, 16 + 8, 16)
        fs2 = fs2.split(0, 16, 8)
        fs2.integrate([1, 1, 1], 0.025)
        fs2 = fs2.pulse_migrate(0, 2, 8, 0.1 / 0.8)
        fs2 = fs2.pulse_migrate(1, 2, 8, 0.2)
        fs2.integrate([1, 1, 1], 0.025)

        self.assertEqual(fs.Npop, 3)
        self.assertEqual(len(fs.shape), len(fs2.shape))
        for n1, n2 in zip(fs.sample_sizes, fs2.sample_sizes):
            self.assertEqual(n1, n2)

        self.assertTrue(np.all(np.isclose(fs, fs2)))

    def test_multimerger(self):
        b = demes.Builder()
        b.add_deme("x", epochs=[dict(start_size=1000, end_time=100)])
        b.add_deme("a", ancestors=["x"], epochs=[dict(start_size=1000, end_time=50)])
        b.add_deme("b", ancestors=["x"], epochs=[dict(start_size=1000, end_time=50)])
        b.add_deme(
            "c",
            ancestors=["a", "b"],
            proportions=[0.5, 0.5],
            start_time=50,
            epochs=[dict(start_size=1000)],
        )
        b.add_deme(
            "d",
            ancestors=["a", "b"],
            proportions=[0.5, 0.5],
            start_time=50,
            epochs=[dict(start_size=1000)],
        )
        graph = b.resolve()

        y_graph = Demes.LD(graph, sampled_demes=["c", "d"], theta=1)

        y = moments.LD.Demographics1D.snm(theta=1)
        y = y.split(0)
        y.integrate([1, 1], 0.025, theta=1)
        y = y.admix(0, 1, 0.5)
        y = y.admix(0, 1, 0.5)
        y = y.marginalize([0, 1])
        y.integrate([1, 1], 0.025, theta=1)

        fs = moments.Spectrum.from_demes(
            graph, sampled_demes=["c", "d"], sample_sizes=[10, 10]
        )
        fs = fs.project([2, 2])

        assert np.allclose(y_graph[0], y[0])
        assert np.isclose(fs.marginalize([1])[1], y[0][0])
        assert np.isclose(fs.marginalize([0])[1], y[0][2])
        assert np.isclose(fs.project([1, 1]).sum(), y[0][1])


class TestFStatistics(unittest.TestCase):
    # f-statistics computed from heterozygosities from moments.LD
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_f2_equivalent(self):
        T = 100
        N = 1000
        b1 = demes.Builder()
        b1.add_deme("A", epochs=[dict(start_size=N)])
        g1 = b1.resolve()
        b2 = demes.Builder()
        b2.add_deme("A", epochs=[dict(start_size=N)])
        b2.add_deme("B", ancestors=["A"], start_time=T, epochs=[dict(start_size=N)])
        g2 = b2.resolve()
        b3 = demes.Builder()
        b3.add_deme("A", epochs=[dict(start_size=1000)])
        b3.add_deme(
            "B",
            ancestors=["A"],
            start_time=3 * T / 2,
            epochs=[dict(start_size=N, end_time=T)],
        )
        g3 = b3.resolve()

        theta = 0.001
        y1 = Demes.LD(
            g1, sampled_demes=["A", "A"], sample_times=[0, 2 * T], theta=theta
        )
        y2 = Demes.LD(g2, sampled_demes=["A", "B"], sample_times=[0, 0], theta=theta)
        y3 = Demes.LD(g3, sampled_demes=["A", "B"], theta=theta)

        assert np.allclose(y1[0], y2[0])
        assert np.allclose(y1[0], y3[0])

        f2 = y2.f2("A", "B")
        t = T / 2 / N
        E_f2 = theta * t
        assert np.isclose(f2, y2.f2(0, 1))
        assert np.isclose(f2, y2.f2("B", "A"))
        assert np.isclose(f2, E_f2)

    def test_f4_ancient_samples(self):
        T = 100
        b = demes.Builder(defaults=dict(epoch=dict(start_size=1000)))
        b.add_deme("anc", epochs=[dict(end_time=T)])
        b.add_deme("A", ancestors=["anc"])
        b.add_deme("B", ancestors=["anc"])
        g = b.resolve()

        theta = 1
        y1 = Demes.LD(
            g,
            sampled_demes=["A", "B", "A", "B"],
            sample_times=[0, 0, T / 2, T / 2],
            theta=theta,
        )
        y2 = Demes.LD(
            g,
            sampled_demes=["A", "B", "A", "B"],
            sample_times=[0, 0, 3 * T / 4, T / 4],
            theta=theta,
        )

        assert np.isclose(y1.f2("A", "B"), y2.f2("A", "B"))
        assert np.isclose(y1.f2(0, 1), y1.f4(0, 1, 0, 1))
        assert np.isclose(y1.f2(0, 1), y1.f4(1, 0, 1, 0))
        assert np.isclose(y1.f4(0, 1, 0, 1), -y2.f4(0, 1, 1, 0))
        assert np.isclose(y1.f4(0, 1, 2, 3), y2.f4(0, 1, 2, 3))
        assert np.isclose(y1.f4(0, 2, 1, 3), y2.f4(2, 0, 3, 1))
        assert np.isclose(y1.f4(0, 1, 2, 3), -y2.f4(0, 1, 3, 2))

    def test_two_population_pulse(self):
        N = 1000

        def build_model(T1, T2, f):
            b = demes.Builder(defaults=dict(epoch=dict(start_size=N)))
            b.add_deme("anc", epochs=[dict(end_time=T1 + T2)])
            b.add_deme("A", ancestors=["anc"])
            b.add_deme("B", ancestors=["anc"])
            b.add_pulse(sources=["B"], dest="A", proportions=[f], time=T2)
            return b.resolve()

        T1 = 100
        T2 = 100
        theta = 1
        for f in [0, 0.1, 0.5, 1.0]:
            g = build_model(T1, T2, f)
            y = Demes.LD(g, sampled_demes=["A", "B"], theta=theta)
            assert np.isclose(
                y.f2(0, 1),
                theta * (T2 + (1 - f) ** 2 * T1) / 2 / N,
                rtol=0.01,
            )

    def test_ancient_structure_loop(self):
        N = 1000

        def build_model(T1, T2, T3, f):
            b = demes.Builder(defaults=dict(epoch=dict(start_size=N)))
            b.add_deme("anc", epochs=[dict(end_time=T1 + T2 + T3)])
            b.add_deme("stem1", ancestors=["anc"], epochs=[dict(end_time=T1 + T2)])
            b.add_deme("stem2", ancestors=["anc"], epochs=[dict(end_time=T1 + T2)])
            b.add_deme(
                "X",
                ancestors=["stem1", "stem2"],
                proportions=[f, 1 - f],
                start_time=T1 + T2,
                epochs=[dict(end_time=T1)],
            )
            b.add_deme("modern1", ancestors=["X"], epochs=[dict(end_time=0)])
            b.add_deme("modern2", ancestors=["X"], epochs=[dict(end_time=0)])
            return b.resolve()

        for f in [0.1, 0.5, 0.8]:
            g = build_model(100, 100, 100, f)
            y = Demes.LD(
                g,
                sampled_demes=["modern1", "modern2", "stem1", "stem2"],
                sample_times=[0, 0, 250, 250],
            )
            assert y.f4(0, 1, 2, 3) == 0.0
            assert y.f4(0, 2, 1, 3) > 0
            assert y.f4(0, 2, 3, 1) < 0

    def test_ancient_structure_admixture(self):
        N = 1000

        def build_model(T1, T2, x, f):
            b = demes.Builder(defaults=dict(epoch=dict(start_size=N)))
            b.add_deme("anc", epochs=[dict(end_time=T1 + T2)])
            b.add_deme("stem1", ancestors=["anc"], epochs=[dict(end_time=0)])
            b.add_deme("stem2", ancestors=["anc"], epochs=[dict(end_time=T1)])
            b.add_deme("modern1", ancestors=["stem2"], epochs=[dict(end_time=0)])
            b.add_deme("modern2", ancestors=["stem2"], epochs=[dict(end_time=0)])
            b.add_pulse(sources=["stem1"], dest="modern1", time=x * T1, proportions=[f])
            return b.resolve()

        T1 = T2 = 1000
        for x in [0.1, 0.2, 0.5, 0.8]:
            for f in [0, 0.1, 0.2, 0.5, 0.8, 1]:
                g = build_model(T1, T2, x, f)
                y = Demes.LD(
                    g,
                    sampled_demes=["modern1", "modern2", "stem1", "stem2"],
                    sample_times=[0, 0, T1 + T2 / 2, T1 + T2 / 2],
                )
                print(x, f, f * y.f2(2, 3), y.f4(0, 1, 2, 3))
                assert np.isclose(f * y.f2(2, 3), y.f4(0, 1, 2, 3))
        g1 = build_model(T1, T2, 0.1, 0.5)
        y1 = Demes.LD(
            g1,
            sampled_demes=["modern1", "modern2", "stem1", "stem2"],
            sample_times=[0, 0, T1 + T2 / 2, T1 + T2 / 2],
        )
        g2 = build_model(T1, T2, 0.9, 0.5)
        y2 = Demes.LD(
            g2,
            sampled_demes=["modern1", "modern2", "stem1", "stem2"],
            sample_times=[0, 0, T1 + T2 / 2, T1 + T2 / 2],
        )
        assert np.isclose(y1.f4(0, 1, 2, 3), y2.f4(0, 1, 2, 3))


class TestSelectionSFS(unittest.TestCase):
    # f-statistics computed from heterozygosities from moments.LD
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_selection_dict_setup(self):
        gamma_dict, h_dict = Demes._set_up_selection_dicts(None, None)
        assert len(gamma_dict) == 1
        assert len(h_dict) == 1
        assert "_default" in gamma_dict
        assert "_default" in h_dict

        gamma_dict, h_dict = Demes._set_up_selection_dicts(None, 1)
        assert len(gamma_dict) == 1
        assert len(h_dict) == 1
        assert "_default" in gamma_dict
        assert "_default" in h_dict

        gamma_dict, h_dict = Demes._set_up_selection_dicts(1, None)
        assert len(gamma_dict) == 1
        assert "_default" in gamma_dict
        assert len(h_dict) == 1
        assert "_default" in h_dict
        assert h_dict["_default"] == 0.5

        gamma_dict, h_dict = Demes._set_up_selection_dicts({"x": 1, "y": 2}, None)
        assert len(gamma_dict) == 3
        assert "x" in gamma_dict
        assert "y" in gamma_dict
        assert "_default" in gamma_dict
        assert len(h_dict) == 1

    def test_single_selection_coefficient_one_pop(self):
        gamma = -1
        n = 30
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))

        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        g = b.resolve()
        fs_demes = Demes.SFS(
            g, sampled_demes=["A"], sample_sizes=[n], gamma=gamma, theta=1
        )

        assert np.allclose(fs, fs_demes)

        fs.integrate([2], 0.1, gamma=gamma)

        b = demes.Builder()
        b.add_deme(
            "A", epochs=[dict(start_size=1000, end_time=200), dict(start_size=2000)]
        )
        g = b.resolve()
        fs_demes = Demes.SFS(
            g, sampled_demes=["A"], sample_sizes=[n], gamma=gamma, theta=1
        )

        assert np.allclose(fs, fs_demes)

    def test_changed_selection_coefficient(self):
        gamma1 = -1
        gamma2 = -2
        n = 30
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma1))
        fs.integrate([1], 0.5, gamma=gamma2)

        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=1000)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()
        fs_demes = Demes.SFS(
            g,
            sampled_demes=["B"],
            sample_sizes=[n],
            gamma={"A": gamma1, "B": gamma2},
            theta=1,
        )

        assert np.allclose(fs, fs_demes)

    def test_changed_dominance_coefficient(self):
        gamma = -1
        h1 = 0.2
        h2 = 0.8
        n = 30
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma, h=h1)
        )
        fs.integrate([1], 0.5, gamma=gamma, h=h2)

        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=1000)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()
        fs_demes = Demes.SFS(
            g,
            sampled_demes=["B"],
            sample_sizes=[n],
            gamma=gamma,
            h={"A": h1, "B": h2},
            theta=1,
        )

        assert np.allclose(fs, fs_demes)

    def test_defaults_one_pop(self):
        gamma1 = -5
        gamma2 = -1
        h1 = 0.2
        h2 = 0
        n = 30
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma1, h=h1)
        )
        fs.integrate([1], 0.5, gamma=gamma2, h=h2)

        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=1000)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()
        fs_demes = Demes.SFS(
            g,
            sampled_demes=["B"],
            sample_sizes=[n],
            gamma={"A": gamma1, "B": gamma2},
            h={"A": h1, "B": h2},
            theta=1,
        )
        fs_demes2 = Demes.SFS(
            g,
            sampled_demes=["B"],
            sample_sizes=[n],
            gamma={"_default": -5, "B": gamma2},
            h={"_default": 0, "A": h1},
            theta=1,
        )

        assert np.allclose(fs, fs_demes)
        assert np.allclose(fs, fs_demes2)

    def test_missing_dict_keys(self):
        gamma1 = 0
        gamma2 = -1
        h1 = 0.2
        h2 = 0.5
        n = 30
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma1, h=h1)
        )
        fs.integrate([1], 0.5, gamma=gamma2, h=h2)

        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=1000)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()
        fs_demes = Demes.SFS(
            g,
            sampled_demes=["B"],
            sample_sizes=[n],
            gamma={"A": gamma1, "B": gamma2},
            h={"A": h1, "B": h2},
            theta=1,
        )
        fs_demes2 = Demes.SFS(
            g,
            sampled_demes=["B"],
            sample_sizes=[n],
            gamma={"B": gamma2},
            h={"A": h1},
            theta=1,
        )

        assert np.allclose(fs, fs_demes)
        assert np.allclose(fs, fs_demes2)

    def test_split_with_variable_coefficients(self):
        gamma0 = -1
        gamma1 = -3
        gamma2 = 5
        h0 = 0.2
        h1 = 0.1
        h2 = 0.95
        n = 30

        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h0)
        )
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], 0.25, gamma=[gamma1, gamma2], h=[h1, h2])

        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=500)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        b.add_deme("C", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()

        fs_demes = Demes.SFS(
            g,
            sampled_demes=["B", "C"],
            sample_sizes=[n, n],
            gamma={"A": gamma0, "B": gamma1, "C": gamma2},
            h={"A": h0, "B": h1, "C": h2},
            theta=1,
        )

        assert np.allclose(fs, fs_demes)

    def test_scalar_vs_defaults(self):
        gamma = -1
        h = 0.2
        n = 30
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000)])
        g = b.resolve()
        fs1 = Demes.SFS(
            g,
            sampled_demes=["A"],
            sample_sizes=[n],
            gamma=gamma,
            h=h,
            theta=1,
        )
        fs2 = Demes.SFS(
            g,
            sampled_demes=["A"],
            sample_sizes=[n],
            gamma={"_default": gamma},
            h={"_default": h},
            theta=1,
        )
        assert np.allclose(fs1, fs2)

    def test_bad_selection(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=500)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        b.add_deme("C", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()

        for gamma in ["hi", ["hi"], {1, 2}, [1, 2], np.inf, np.nan]:
            with self.assertRaises(ValueError):
                Demes.SFS(g, ["B", "C"], [10, 10], gamma=gamma, theta=1)

        for gamma in [{"D": 1}]:
            with self.assertRaises(ValueError):
                Demes.SFS(g, ["B", "C"], [10, 10], gamma=gamma, theta=1)

    def test_bad_dominance(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=500)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        b.add_deme("C", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()

        for h in ["hi", ["hi"], {1, 2}, [1, 2]]:
            with self.assertRaises(TypeError):
                Demes.SFS(g, ["B", "C"], [10, 10], h=h, theta=1)

        for g in [{"D": 1}]:
            with self.assertRaises(ValueError):
                Demes.SFS(g, ["B", "C"], [10, 10], h=h, theta=1)

    def test_bad_graph_with_selection(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=500)])
        b.add_deme("B", ancestors=["A"], epochs=[dict(start_size=1000)])
        b.add_deme("_default", ancestors=["A"], epochs=[dict(start_size=1000)])
        g = b.resolve()

        with self.assertRaises(ValueError):
            Demes.SFS(g, ["B"], [10], gamma=-1, theta=1)

        with self.assertRaises(ValueError):
            Demes.SFS(g, ["B"], [10], gamma={"_default": -1}, theta=1)

        Demes.SFS(g, ["B"], [10], h=0.1, theta=1)


class TestSamplesSpecification(unittest.TestCase):
    # f-statistics computed from heterozygosities from moments.LD
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_samples_equivalence(self):
        graph = demes.load(
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yaml")
        )
        fs1 = moments.Spectrum.from_demes(
            graph, samples={"YRI": 8, "CEU": 10, "CHB": 12}
        )
        fs2 = moments.Spectrum.from_demes(
            graph, sampled_demes=["YRI", "CEU", "CHB"], sample_sizes=[8, 10, 12]
        )
        assert np.allclose(fs1, fs2)

    def test_bad_samples(self):
        graph = demes.load(
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yaml")
        )

        with self.assertRaises(ValueError):
            samples = {"JPT": 10}
            fs = moments.Spectrum.from_demes(graph, samples=samples)
        with self.assertRaises(ValueError):
            samples = {"CEU": 8}
            fs = moments.Spectrum.from_demes(graph, samples=samples, sample_times=[1])
        with self.assertRaises(ValueError):
            samples = {"CEU": 10}
            fs = moments.Spectrum.from_demes(
                graph, samples=samples, sampled_demes=["YRI"], sample_sizes=[10]
            )


class TestAncientSamples(unittest.TestCase):
    """
    We want to be able to draw all ancient samples, and not simulate all the
    way to time zero.

    This includes if we sample in the ancient past while the more recent history
    has more than 5 populations. It should be allowed
    """

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_slice_times(self):
        g = demes.load(
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yaml")
        )
        for t in [-1, -math.inf, math.inf]:
            with self.assertRaises(ValueError):
                moments.Demes.DemesUtil.slice(g, t)
        for t in [-1, 0, -math.inf, math.inf]:
            with self.assertRaises(ValueError):
                moments.Demes.DemesUtil.swipe(g, t)

    def test_single_population_slice(self):
        b = demes.Builder()
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=100, end_time=50),
                dict(start_size=200, end_time=0),
            ],
        )
        g = b.resolve()

        g2 = moments.Demes.DemesUtil.slice(g, 10)
        self.assertEqual(g2["a"].epochs[0].end_time, 40)
        self.assertEqual(g2["a"].epochs[1].end_time, 0)
        self.assertEqual(g2["a"].epochs[1].start_size, 200)

        g3 = moments.Demes.DemesUtil.slice(g, 100)
        self.assertEqual(len(g3["a"].epochs), 1)
        self.assertEqual(g3["a"].epochs[0].start_size, 100)

        g4 = moments.Demes.DemesUtil.slice(g, 50)
        self.assertEqual(len(g4["a"].epochs), 1)
        self.assertEqual(g4["a"].epochs[0].start_size, 100)

    def test_single_population_exponential_slice(self):
        b = demes.Builder()
        start_size = 200
        end_size = 500
        T = 50
        b.add_deme(
            "a",
            epochs=[
                dict(start_size=100, end_time=T),
                dict(start_size=start_size, end_size=end_size, end_time=0),
            ],
        )
        g = b.resolve()

        t = 10
        g2 = moments.Demes.DemesUtil.slice(g, t)
        self.assertEqual(
            g2["a"].epochs[1].end_size,
            start_size * math.exp(math.log(end_size / start_size) * (T - t) / T),
        )

    def test_slice_integration(self):
        b = demes.Builder()
        b.add_deme(name="ancestral", epochs=[dict(start_size=2000, end_time=1000)])
        b.add_deme(
            name="deme1",
            ancestors=["ancestral"],
            epochs=[dict(start_size=1500, end_size=1000)],
        )
        b.add_deme(
            name="deme2",
            ancestors=["ancestral"],
            epochs=[dict(start_size=500, end_size=3000)],
        )
        g = b.resolve()

        # test that they run properly
        for t in [0, 1, 10]:
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[0, t])
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[t, t])
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[t, 2 * t])

        b.add_migration(demes=["deme1", "deme2"], rate=2e-3)
        g = b.resolve()

        for t in [0, 1, 10]:
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[0, t])
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[t, t])
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[t, 2 * t])

        b.add_pulse(sources=["deme1"], dest="deme2", proportions=[0.1], time=200)
        g = b.resolve()

        for t in [0, 1, 10]:
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[0, t])
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[t, t])
            y = Demes.LD(g, sampled_demes=["deme1", "deme2"], sample_times=[t, 2 * t])

    def test_slice_results(self):
        b = demes.Builder()
        b.add_deme(name="ancestral", epochs=[dict(start_size=2000, end_time=1000)])
        b.add_deme(
            name="deme1", ancestors=["ancestral"], epochs=[dict(start_size=1000)]
        )
        b.add_deme(
            name="deme2", ancestors=["ancestral"], epochs=[dict(start_size=4000)]
        )
        b.add_migration(demes=["deme1", "deme2"], rate=2e-3)
        b.add_pulse(sources=["deme1"], dest="deme2", proportions=[0.1], time=500)
        g = b.resolve()

        # test some results
        theta = 0.001
        y = Demes.LD(
            g, sampled_demes=["deme1", "deme2"], sample_times=[100, 200], theta=theta
        )
        y2 = moments.LD.Demographics1D.snm(theta=theta)
        y2 = y2.split(0)
        y2.integrate([0.5, 2.0], 0.125, m=[[0, 8], [8, 0]], theta=theta)
        y2 = y2.pulse_migrate(0, 1, 0.1)
        y2.integrate([0.5, 2.0], 0.075, m=[[0, 8], [8, 0]], theta=theta)
        y2 = y2.split(1)
        y2.integrate(
            [0.5, 2.0, 2.0],
            0.025,
            m=[[0, 8, 0], [8, 0, 0], [0, 0, 0]],
            frozen=[False, False, True],
            theta=theta,
        )
        y2 = y2.marginalize([1])
        self.assertTrue(np.allclose(y[0], y2[0]))

    def test_multipopulation_slice(self):
        b = demes.Builder()
        b.add_deme(name="ancestral", epochs=[dict(start_size=2000, end_time=1000)])
        b.add_deme(
            name="deme1", ancestors=["ancestral"], epochs=[dict(start_size=1000)]
        )
        b.add_deme(
            name="deme2", ancestors=["ancestral"], epochs=[dict(start_size=4000)]
        )
        b.add_deme(
            name="deme3",
            ancestors=["deme1"],
            start_time=500,
            epochs=[dict(start_size=5000)],
        )
        b.add_migration(demes=["deme1", "deme2"], rate=1e-3)
        b.add_migration(
            demes=["deme1", "deme3"], start_time=400, end_time=100, rate=2e-3
        )
        b.add_migration(source="deme2", dest="deme3", rate=1e-4)
        b.add_pulse(sources=["deme3"], dest="deme1", proportions=[0.1], time=300)
        g = b.resolve()

        g2 = moments.Demes.DemesUtil.slice(g, 50)
        self.assertEqual(len(g2.migrations), 5)
        self.assertEqual(len(g2.pulses), 1)
        self.assertEqual(g2.migrations[2].end_time, 50)
        self.assertEqual(g2.pulses[0].time, 250)

        g3 = moments.Demes.DemesUtil.slice(g, 350)
        self.assertEqual(len(g3.migrations), 5)
        self.assertEqual(len(g3.pulses), 0)
        self.assertEqual(g3.migrations[2].end_time, 0)
        self.assertEqual(g3.migrations[2].start_time, 50)

        g4 = moments.Demes.DemesUtil.slice(g, 500)
        self.assertEqual(len(g4.migrations), 2)
        self.assertEqual(len(g4.pulses), 0)
        self.assertEqual(len(g4.demes), 3)

    def test_multipopulation_swipe(self):
        b = demes.Builder()
        b.add_deme(name="ancestral", epochs=[dict(start_size=2000, end_time=1000)])
        b.add_deme(
            name="deme1", ancestors=["ancestral"], epochs=[dict(start_size=1000)]
        )
        b.add_deme(
            name="deme2", ancestors=["ancestral"], epochs=[dict(start_size=4000)]
        )
        b.add_deme(
            name="deme3",
            ancestors=["deme1"],
            start_time=500,
            epochs=[dict(start_size=5000)],
        )
        b.add_migration(demes=["deme1", "deme2"], rate=1e-3)
        b.add_migration(
            demes=["deme1", "deme3"], start_time=400, end_time=100, rate=2e-3
        )
        b.add_migration(source="deme2", dest="deme3", rate=1e-4)
        b.add_pulse(sources=["deme3"], dest="deme1", proportions=[0.1], time=300)
        g = b.resolve()

        g2 = moments.Demes.DemesUtil.swipe(g, 50)
        self.assertEqual(len(g2.migrations), 3)
        self.assertEqual(len(g2.pulses), 0)
        self.assertEqual(g2.migrations[0].start_time, 50)
        self.assertEqual(g2.migrations[1].start_time, 50)
        self.assertEqual(g2.migrations[2].start_time, 50)
        self.assertEqual(len(g2.demes), 3)

        g3 = moments.Demes.DemesUtil.swipe(g, 350)
        self.assertEqual(len(g3.migrations), 5)
        self.assertEqual(len(g3.pulses), 1)
        self.assertEqual(g3.migrations[2].end_time, 100)
        self.assertEqual(g3.migrations[3].end_time, 100)
        self.assertEqual(len(g3.demes), 3)

        g4 = moments.Demes.DemesUtil.swipe(g, 500)
        self.assertEqual(len(g4.demes), 3)
        self.assertEqual(g4.demes[2].ancestors[0], "deme1")

        g5 = moments.Demes.DemesUtil.swipe(g, 1000)
        self.assertEqual(g, g5)

        g6 = moments.Demes.DemesUtil.swipe(g, 2000)
        self.assertFalse(g == g6)

    def test_ancient_samples(self):
        # test SFS inference with many pops in recent time but ancient samples
        b = demes.Builder()
        b.add_deme(
            "ancestral",
            epochs=[
                dict(start_size=100, end_time=300),
                dict(start_size=200, end_time=100),
            ],
        )
        for d in ["A", "B", "C", "D", "E", "F", "G"]:
            b.add_deme(
                d, ancestors=["ancestral"], epochs=[dict(start_size=100, end_time=0)]
            )
        g = b.resolve()

        with self.assertRaises(ValueError):
            sfs = moments.Spectrum.from_demes(g, samples={"A": 10})
        with self.assertRaises(ValueError):
            sfs = moments.Spectrum.from_demes(
                g, sampled_demes=["A"], sample_sizes=[10], sample_times=[10]
            )

        sfs = moments.Spectrum.from_demes(
            g, sampled_demes=["ancestral"], sample_sizes=[10], sample_times=[200]
        )
        sfs2 = moments.Demographics1D.snm([10])
        sfs2.integrate([2], 0.5)
        self.assertTrue(np.allclose(sfs.data, sfs2.data))


class TestSFSScaling(unittest.TestCase):
    # Scaling of the SFS
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_mutation_rates(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=5678)])
        g = b.resolve()
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], theta=[1])
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], theta=0)
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], theta=-1)
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], theta=1, u=1)
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], theta=1, u=1e-8)
        with self.assertRaises(ValueError):
            Demes.SFS(
                g, sampled_demes=["A"], sample_sizes=[4], theta=1, reversible=True
            )
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], reversible=True)
        with self.assertRaises(ValueError):
            Demes.SFS(g, sampled_demes=["A"], sample_sizes=[4], reversible=True, u=1)
        with self.assertRaises(ValueError):
            Demes.SFS(
                g,
                sampled_demes=["A"],
                sample_sizes=[4],
                reversible=True,
                theta=[[0.1, 0.1]],
            )
        with self.assertRaises(ValueError):
            Demes.SFS(
                g,
                sampled_demes=["A"],
                sample_sizes=[4],
                reversible=True,
                u=[[1e-8, 1e-8]],
            )
        with self.assertRaises(ValueError):
            Demes.SFS(
                g, sampled_demes=["A"], sample_sizes=[4], reversible=True, u=1e-8, L=2
            )

    def test_default_theta(self):
        # default scaling of SFS should be with theta = 1
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=5678)])
        g = b.resolve()
        n = 100
        fs_demes = moments.Spectrum.from_demes(g, samples={"A": n})
        fs_demes2 = Demes.SFS(g, sampled_demes=["A"], sample_sizes=[n], theta=1)
        fs = moments.Demographics1D.snm([n])
        self.assertTrue(np.allclose(fs, fs_demes))
        self.assertTrue(np.allclose(fs, fs_demes2))

    def test_theta_not_one(self):
        # default scaling of SFS should be with theta = 1
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=5678)])
        g = b.resolve()
        n = 100
        theta = 1000
        fs_demes = moments.Spectrum.from_demes(g, samples={"A": n}, theta=theta)
        fs_demes2 = Demes.SFS(g, sampled_demes=["A"], sample_sizes=[n], theta=theta)
        fs = moments.Demographics1D.snm([n]) * theta
        self.assertTrue(np.allclose(fs, fs_demes))
        self.assertTrue(np.allclose(fs, fs_demes2))

    def test_no_theta_given(self):
        b = demes.Builder()
        Ne = 15000
        b.add_deme("A", epochs=[dict(start_size=Ne)])
        g = b.resolve()
        n = 20
        fs = moments.Spectrum.from_demes(g, samples={"A": n})
        fs2 = moments.Demographics1D.snm([n])
        self.assertTrue(np.allclose(fs, fs2))
        fs = Demes.SFS(g, samples={"A": n})
        fs2 = moments.Demographics1D.snm([n]) * 4 * Ne
        self.assertTrue(np.allclose(fs, fs2))

    def test_size_change(self):
        b = demes.Builder()
        Ne = 10000
        b.add_deme(
            "A", epochs=[dict(start_size=Ne, end_time=2000), dict(start_size=2 * Ne)]
        )
        g = b.resolve()
        n = 40
        fs = Demes.SFS(g, samples={"A": n}, theta=1)
        fs2 = moments.Demographics1D.two_epoch([2, 0.1], [n])
        self.assertTrue(np.allclose(fs, fs2))
        fs = Demes.SFS(g, samples={"A": n})
        fs2 = moments.Demographics1D.two_epoch([2, 0.1], [n]) * 4 * Ne
        self.assertTrue(np.allclose(fs, fs2))

    def test_two_pops(self):
        b = demes.Builder()
        Ne = 10000
        b.add_deme("anc", epochs=[dict(start_size=Ne, end_time=2000)])
        b.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=2 * Ne)])
        b.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=3 * Ne)])
        g = b.resolve()
        n = 20
        fs = Demes.SFS(g, samples={"A": n, "B": n})
        fs2 = moments.Demographics2D.split_mig([2, 3, 0.1, 0], [n, n]) * 4 * Ne
        self.assertTrue(np.allclose(fs, fs2))

    def test_reversible(self):
        b = demes.Builder()
        Ne = 10000
        b.add_deme(
            "A",
            epochs=[
                dict(start_size=Ne, end_time=0.1 * 2 * Ne),
                dict(start_size=2 * Ne),
            ],
        )
        g = b.resolve()
        n = 40
        u = [2e-8, 1e-8]
        gamma = -1
        fs = Demes.SFS(g, samples={"A": n}, reversible=True, u=u, gamma=gamma)
        self.assertFalse(np.any(fs.mask))
        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                n, gamma=gamma, theta_fd=4 * Ne * u[0], theta_bd=4 * Ne * u[1]
            ),
            mask_corners=False,
        )
        fs2.integrate(
            [2],
            0.1,
            gamma=gamma,
            theta_fd=4 * Ne * u[0],
            theta_bd=4 * Ne * u[1],
            finite_genome=True,
        )
        self.assertTrue(np.allclose(fs, fs2))

    def test_reversible_mutation_rates(self):
        b = demes.Builder()
        Ne = 10000
        b.add_deme(
            "A", epochs=[dict(start_size=Ne, end_time=1000), dict(start_size=5 * Ne)]
        )
        g = b.resolve()
        n = 30
        u = 1e-7
        fs = Demes.SFS(g, samples={"A": n}, reversible=True, u=u)
        fs2 = Demes.SFS(g, samples={"A": n}, reversible=True, u=[u, u])
        fs3 = Demes.SFS(g, samples={"A": n}, reversible=True, u=u, L=1)
        self.assertTrue(np.allclose(fs, fs2))
        self.assertTrue(np.allclose(fs, fs3))


class TestDemesRescaling(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_scaling_value(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1)])
        g = b.resolve()
        with self.assertRaises(ValueError):
            moments.Demes.DemesUtil.rescale(g, -1)
        with self.assertRaises(ValueError):
            moments.Demes.DemesUtil.rescale(g, 0)
        with self.assertRaises(ValueError):
            moments.Demes.DemesUtil.rescale(g, np.inf)

    def test_rescaling_reversal(self):
        b = demes.Builder()
        b.add_deme(
            "anc",
            epochs=[
                dict(start_size=100, end_time=100),
                dict(start_size=200, end_size=300, end_time=50),
            ],
        )
        b.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=500)])
        b.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=100, end_size=10)])
        b.add_migration(demes=["A", "B"], rate=0.01)
        b.add_pulse(sources=["A"], dest="B", proportions=[0.1], time=20)
        b.add_pulse(sources=["B"], dest="A", proportions=[0.2], time=10)
        g = b.resolve()

        g2 = moments.Demes.DemesUtil.rescale(g, 1)
        g.assert_close(g2)

        g3 = moments.Demes.DemesUtil.rescale(g, 0.1)
        g4 = moments.Demes.DemesUtil.rescale(g3, 10)
        g.assert_close(g4)

        g5 = moments.Demes.DemesUtil.rescale(g3, 0.2)
        g6 = moments.Demes.DemesUtil.rescale(g5, 50)
        g.assert_close(g6)

    def IM_model(self, Ne=10000):
        b = demes.Builder()
        b.add_deme(
            "anc",
            epochs=[
                dict(start_size=Ne, end_time=0.1 * 2 * Ne),
            ],
        )
        b.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=0.5 * Ne)])
        b.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=2 * Ne)])
        return b

    def test_rescaled_mutation_rate(self):
        Ne = 10000
        g = self.IM_model(Ne=Ne).resolve()
        samples = {"A": 30, "B": 30}
        u = 1e-8
        L = 1e4
        fs = moments.Demes.SFS(g, samples=samples, u=u, L=L)
        fsb = moments.Demes.SFS(g, samples=samples, theta=4 * Ne * u * L)
        self.assertTrue(np.allclose(fs, fsb))
        fsc = moments.Demes.SFS(g, samples=samples, theta=4 * Ne * u, L=L)
        self.assertTrue(np.allclose(fs, fsc))

        for Q in [10, 2, 0.1]:
            g2 = moments.Demes.DemesUtil.rescale(g, Q)
            fs2 = moments.Demes.SFS(g2, samples=samples, u=u * Q, L=L)
            self.assertTrue(np.allclose(fs, fs2))
            fs3 = moments.Demes.SFS(g2, samples=samples, theta=4 * Ne * u, L=L)
            self.assertTrue(np.allclose(fs, fs3))

    def test_rescaled_with_migration(self):
        b = self.IM_model(Ne=10000)
        b.add_migration(source="A", dest="B", rate=1e-4)
        b.add_migration(source="B", dest="A", rate=2e-4, start_time=2000, end_time=1000)
        g = b.resolve()

        samples = {"A": 30, "B": 30}
        u = 1e-8
        fs = moments.Demes.SFS(g, samples=samples, u=u)

        for Q in [10, 2, 0.5]:
            g2 = moments.Demes.DemesUtil.rescale(g, Q)
            fs2 = moments.Demes.SFS(g2, samples=samples, u=u * Q)
            self.assertTrue(np.allclose(fs, fs2))

    def test_rescaled_reversible_mutation(self):
        Ne = 10000
        g = self.IM_model(Ne=Ne).resolve()
        samples = {"A": 30, "B": 30}
        u1 = 1e-8
        u2 = 2e-8
        u = [u1, u2]
        theta = [4 * Ne * u1, 4 * Ne * u2]
        fs = moments.Demes.SFS(g, samples=samples, u=u, reversible=True)
        fsb = moments.Demes.SFS(g, samples=samples, theta=theta, reversible=True)

        for Q in [10, 2, 0.5]:
            g2 = moments.Demes.DemesUtil.rescale(g, Q)
            u_scaled = [u1 * Q, u2 * Q]
            fs2 = moments.Demes.SFS(g2, samples=samples, u=u_scaled, reversible=True)
            self.assertTrue(np.allclose(fs, fs2))


class TestDemesSelectionCoefficients(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_inputs(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=10000)])
        g = b.resolve()
        samples = {"A": 40}
        with self.assertRaises(ValueError):
            moments.Demes.SFS(g, samples=samples, gamma=[-1])
        with self.assertRaises(ValueError):
            moments.Demes.SFS(g, samples=samples, gamma=np.inf)
        with self.assertRaises(ValueError):
            moments.Demes.SFS(g, samples=samples, s=[0.0001])
        with self.assertRaises(ValueError):
            moments.Demes.SFS(g, samples=samples, s=np.inf)

    def test_selection_one_pop(self):
        b = demes.Builder()
        Ne = 1e4
        b.add_deme("A", epochs=[dict(start_size=Ne)])
        g = b.resolve()
        samples = {"A": 40}
        s = -1e-4
        gamma = 2 * Ne * s

        fs1 = moments.Demes.SFS(g, samples=samples, gamma=gamma)
        fs2 = moments.Demes.SFS(g, samples=samples, s=s)
        self.assertTrue(np.allclose(fs1, fs2))

        s2 = {"A": s}
        fs3 = moments.Demes.SFS(g, samples=samples, s=s2)
        self.assertTrue(np.allclose(fs1, fs3))


class TestArchaicSamples(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_nonzero_end_time(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1e4)])
        b.add_deme(
            "B",
            ancestors=["A"],
            start_time=2000,
            epochs=[dict(start_size=1e3, end_time=500)],
        )
        g = b.resolve()

        y = moments.Demes.LD(g, sampled_demes=["A", "B"], u=1e-8, r=1e-4)

        b2 = demes.Builder()
        b2.add_deme("A", epochs=[dict(start_size=1e4)])
        b2.add_deme(
            "B", ancestors=["A"], start_time=2000, epochs=[dict(start_size=1e3)]
        )
        g2 = b2.resolve()

        y2 = moments.Demes.LD(
            g, sampled_demes=["A", "B"], sample_times=[0, 500], u=1e-8, r=1e-4
        )

        self.assertTrue(np.allclose(y.H(), y2.H()))
        self.assertTrue(np.allclose(y.LD(), y2.LD()))

    def test_simultaneous_sample_pulse(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1e4)])
        b.add_deme(
            "B",
            ancestors=["A"],
            start_time=2000,
            epochs=[dict(start_size=1e3, end_time=500)],
        )
        b.add_pulse(sources=["B"], dest="A", time=500, proportions=[0.05])
        g = b.resolve()

        y = moments.Demes.LD(g, sampled_demes=["A"])

        y2 = moments.Demes.LD(g, sampled_demes=["A", "B"])
