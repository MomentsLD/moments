import os
import unittest
import math
import pathlib

import numpy as np
import moments
from moments.Demes import Demes
import time

import demes


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
        out = Demes._apply_LD_events(y, ("split", "O", pop_ids), 0, pop_ids)
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
        out = Demes._apply_LD_events(y, ("split", "B", child_ids), 0, ["A"] + child_ids)
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
        out = Demes._apply_LD_events(y, ("split", "A", child_ids), 0, ["B"] + child_ids)
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
        fs = moments.Demographics2D.snm([20, 10], pop_ids=["A", "B"])
        out = Demes._pulse_fs(fs, "A", "B", 0.2, [10, 10])
        self.assertTrue(out.sample_sizes[0] == 10)
        self.assertTrue(out.sample_sizes[1] == 10)
        fs2 = fs.pulse_migrate(0, 1, 10, 0.2)
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
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yml"),
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
            theta=theta
        )
        y_demes = moments.LD.LDstats.from_demes(
            os.path.join(os.path.dirname(__file__), "test_files/gutenkunst_ooa.yml"),
            ["YRI", "CEU", "CHB"],
            rho=rho,
            theta=theta,
        )

        self.assertTrue(
            np.all([x == y for x, y in zip(y_moments.pop_ids, y_demes.pop_ids)])
        )
        for x, y in zip(y_demes, y_moments):
            self.assertTrue(np.allclose(x, y))


### tests from demes


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
    # test function operations
    def test_convert_to_generations(self):
        pass

    def test_num_lineages(self):
        # simple merge model
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", initial_size=100, end_time=100)
        g.deme(id="pop1", initial_size=100, ancestors=["anc"], end_time=10)
        g.deme(id="pop2", initial_size=100, ancestors=["anc"], end_time=10)
        g.deme(id="pop3", initial_size=100, ancestors=["anc"], end_time=10)
        g.deme(
            id="pop",
            initial_size=100,
            ancestors=["pop1", "pop2", "pop3"],
            proportions=[0.1, 0.2, 0.7],
            start_time=10,
            end_time=0,
        )
        sampled_demes = ["pop"]
        demes_demo_events = g.list_demographic_events()
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
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", initial_size=100, end_time=100)
        g.deme(id="pop1", initial_size=100, ancestors=["anc"])
        g.deme(id="pop2", initial_size=100, ancestors=["anc"])
        g.deme(id="pop3", initial_size=100, ancestors=["anc"])
        g.deme(
            id="pop",
            initial_size=100,
            ancestors=["pop1", "pop2", "pop3"],
            proportions=[0.1, 0.2, 0.7],
            start_time=10,
            end_time=0,
        )
        sampled_demes = ["pop"]
        demes_demo_events = g.list_demographic_events()
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
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Pop", initial_size=1000)
        fs = Demes.SFS(g, ["Pop"], [20])
        fs_m = moments.Demographics1D.snm([20])
        self.assertTrue(np.allclose(fs.data, fs_m.data))

        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=2000),
                demes.Epoch(end_time=0, initial_size=10000),
            ],
        )
        fs = Demes.SFS(g, ["Pop"], [20])
        fs_m = moments.Demographics1D.snm([20])
        fs_m.integrate([10], 1)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_more_than_5_demes(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", initial_size=1000, end_time=1000)
        for i in range(6):
            g.deme(id=f"pop{i}", initial_size=1000, ancestors=["anc"])
        with self.assertRaises(ValueError):
            Demes.SFS(g, ["pop{i}" for i in range(6)], [10 for i in range(6)])

        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", initial_size=1000, end_time=1000)
        for i in range(3):
            g.deme(id=f"pop{i}", initial_size=1000, ancestors=["anc"])
        with self.assertRaises(ValueError):
            Demes.SFS(
                g,
                ["pop{i}" for i in range(3)],
                [10 for i in range(3)],
                sample_times=[5, 10, 15],
            )

    def test_one_pop_ancient_samples(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Pop", initial_size=1000)
        fs = Demes.SFS(g, ["Pop", "Pop"], [20, 4], sample_times=[0, 100])
        fs_m = moments.Demographics1D.snm([24])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 4)
        fs_m.integrate([1, 1], 100 / 2 / 1000, frozen=[False, True])
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_merge(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Anc", initial_size=1000, end_time=100)
        g.deme(id="Source1", initial_size=2000, ancestors=["Anc"], end_time=10)
        g.deme(id="Source2", initial_size=3000, ancestors=["Anc"], end_time=10)
        g.deme(
            id="Pop",
            initial_size=4000,
            ancestors=["Source1", "Source2"],
            proportions=[0.8, 0.2],
            start_time=10,
        )
        fs = Demes.SFS(g, ["Pop"], [20])

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m.integrate([2, 3], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 20, 0.8)
        fs_m.integrate([4], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_admixture(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="Anc", initial_size=1000, end_time=100)
        g.deme(id="Source1", initial_size=2000, ancestors=["Anc"])
        g.deme(id="Source2", initial_size=3000, ancestors=["Anc"])
        g.deme(
            id="Pop",
            initial_size=4000,
            ancestors=["Source1", "Source2"],
            proportions=[0.8, 0.2],
            start_time=10,
        )
        fs = Demes.SFS(g, ["Source1", "Source2", "Pop"], [10, 10, 10])

        fs_m = moments.Demographics1D.snm([40])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 20, 20)
        fs_m.integrate([2, 3], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.8)
        fs_m.integrate([2, 3, 4], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_growth_models(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(end_time=1000, initial_size=1000),
                demes.Epoch(initial_size=500, final_size=5000, end_time=0),
            ],
        )
        fs = Demes.SFS(g, ["Pop"], [100])

        fs_m = moments.Demographics1D.snm([100])

        def nu_func(t):
            return [0.5 * np.exp(np.log(5000 / 500) * t / 0.5)]

        fs_m.integrate(nu_func, 0.5)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

        g = demes.Graph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(end_time=1000, initial_size=1000),
                demes.Epoch(
                    initial_size=500,
                    final_size=5000,
                    end_time=0,
                    size_function="linear",
                ),
            ],
        )
        fs = Demes.SFS(g, ["Pop"], [100])

        fs_m = moments.Demographics1D.snm([100])

        def nu_func(t):
            return [0.5 + t / 0.5 * (5 - 0.5)]

        fs_m.integrate(nu_func, 0.5)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_simple_pulse_model(self):
        g = demes.Graph(description="test", time_units="generations")
        g.deme(id="anc", initial_size=1000, end_time=100)
        g.deme(id="source", initial_size=1000, ancestors=["anc"])
        g.deme(id="dest", initial_size=1000, ancestors=["anc"])
        g.pulse(source="source", dest="dest", time=10, proportion=0.1)
        fs = Demes.SFS(g, ["source", "dest"], [20, 20])

        fs_m = moments.Demographics1D.snm([60])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 40, 20)
        fs_m.integrate([1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_inplace(fs_m, 0, 1, 20, 0.1)
        fs_m.integrate([1, 1], 10 / 2 / 1000)
        self.assertTrue(np.allclose(fs.data, fs_m.data))

    def test_n_way_split(self):
        g = demes.Graph(description="three-way", time_units="generations")
        g.deme(id="anc", initial_size=1000, end_time=10)
        g.deme(id="deme1", initial_size=1000, ancestors=["anc"])
        g.deme(id="deme2", initial_size=1000, ancestors=["anc"])
        g.deme(id="deme3", initial_size=1000, ancestors=["anc"])
        ns = [10, 15, 20]
        fs = Demes.SFS(g, ["deme1", "deme2", "deme3"], ns)
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
        g = demes.Graph(description="three-way merge", time_units="generations")
        g.deme(id="anc", initial_size=1000, end_time=100)
        g.deme(id="source1", initial_size=1000, end_time=10, ancestors=["anc"])
        g.deme(id="source2", initial_size=1000, end_time=10, ancestors=["anc"])
        g.deme(id="source3", initial_size=1000, end_time=10, ancestors=["anc"])
        g.deme(
            id="merged",
            initial_size=1000,
            ancestors=["source1", "source2", "source3"],
            proportions=[0.5, 0.2, 0.3],
            start_time=10,
        )
        ns = [10]
        fs = Demes.SFS(g, ["merged"], ns)

        fs_m = moments.Demographics1D.snm([30])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 10, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.3)
        fs_m.integrate([1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs_m.data, fs.data))

        g = demes.Graph(description="three-way admix", time_units="generations")
        g.deme(id="anc", initial_size=1000, end_time=100)
        g.deme(id="source1", initial_size=1000, ancestors=["anc"])
        g.deme(id="source2", initial_size=1000, ancestors=["anc"])
        g.deme(id="source3", initial_size=1000, ancestors=["anc"])
        g.deme(
            id="admixed",
            initial_size=1000,
            ancestors=["source1", "source2", "source3"],
            proportions=[0.5, 0.2, 0.3],
            start_time=10,
        )
        ns = [10]
        fs = Demes.SFS(g, ["admixed"], ns)

        fs_m = moments.Demographics1D.snm([30])
        fs_m = moments.Manips.split_1D_to_2D(fs_m, 10, 20)
        fs_m = moments.Manips.split_2D_to_3D_2(fs_m, 10, 10)
        fs_m.integrate([1, 1, 1], 90 / 2 / 1000)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.5 / 0.7)
        fs_m = moments.Manips.admix_into_new(fs_m, 0, 1, 10, 0.3)
        fs_m.integrate([1], 10 / 2 / 1000)

        self.assertTrue(np.allclose(fs_m.data[1:-1], fs.data[1:-1]))

        fs = Demes.SFS(g, ["source1", "admixed"], [10, 10])

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
