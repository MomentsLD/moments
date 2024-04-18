import os
import unittest

import numpy as np
import scipy.special
import moments
import moments.LD
import moments.TwoLocus
import pickle
import time
import copy
import demes


class LDTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_steady_state_fs(self):
        theta = 0.001
        fs = moments.Demographics1D.snm([20]) * theta
        y = moments.LD.Demographics1D.snm(theta=theta)
        self.assertTrue(np.allclose(y[-1][0], fs.project([2])))
        y = y.split(0)
        fs = moments.Manips.split_1D_to_2D(fs, 10, 10)
        fs_proj = fs.project([1, 1])
        self.assertTrue(np.allclose(y[-1][1], fs_proj[0, 1] + fs_proj[1, 0]))

    def test_migration_symmetric_2D(self):
        theta = 0.001
        fs = moments.Demographics1D.snm([30]) * theta
        y = moments.LD.Demographics1D.snm(theta=theta)
        m = 1.0
        T = 0.3
        y = y.split(0)
        y.integrate([1, 1], T, m=[[0, m], [m, 0]], theta=theta)
        fs = moments.Manips.split_1D_to_2D(fs, 15, 15)
        fs.integrate([1, 1], T, m=[[0, m], [m, 0]], theta=theta)
        fs_proj = fs.project([1, 1])
        self.assertTrue(np.allclose(y[-1][1], fs_proj[0, 1] + fs_proj[1, 0], rtol=1e-3))

    def test_migration_asymmetric_2D(self):
        theta = 0.001
        fs = moments.Demographics1D.snm([60]) * theta
        y = moments.LD.Demographics1D.snm(theta=theta)
        m12 = 10.0
        m21 = 0.0
        T = 2.0
        y = y.split(0)
        y.integrate([1, 1], T, m=[[0, m12], [m21, 0]], theta=theta)
        fs = moments.Manips.split_1D_to_2D(fs, 30, 30)
        fs.integrate([1, 1], T, m=[[0, m12], [m21, 0]], theta=theta)
        fs_proj = fs.project([1, 1])
        self.assertTrue(np.allclose(y[-1][1], fs_proj[0, 1] + fs_proj[1, 0], rtol=1e-3))
        fs_proj = fs.marginalize([1]).project([2])
        self.assertTrue(np.allclose(y[-1][0], fs_proj[1], rtol=1e-3))
        fs_proj = fs.marginalize([0]).project([2])
        self.assertTrue(np.allclose(y[-1][2], fs_proj[1], rtol=1e-3))

    def test_equilibrium_ld_tlfs_cache(self):
        theta = 1
        rhos = [0, 1, 10]
        y = moments.LD.Demographics1D.snm(theta=theta, rho=rhos)
        ns = 30
        for ii, rho in enumerate(rhos):
            F = moments.TwoLocus.Demographics.equilibrium(ns, rho=rho).project(4)
            self.assertTrue(np.allclose(y[ii], [F.D2(), F.Dz(), F.pi2()], rtol=5e-2))


class SplitStats(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_split_het(self):
        h = [1]
        h_split_1 = moments.LD.Numerics.split_h(h, 0, 1)
        self.assertEqual(len(h_split_1), 3)
        self.assertTrue(np.all([x == 1 for x in h_split_1]))
        h_split_2 = moments.LD.Numerics.split_h(h_split_1, 1, 2)
        self.assertEqual(len(h_split_2), 6)
        self.assertTrue(np.all([x == 1 for x in h_split_2]))

    def test_split_ld(self):
        y = [1, 2, 3]
        y_split_1 = moments.LD.Numerics.split_ld(y, 0, 1)
        self.assertEqual(len(y_split_1), len(moments.LD.Util.moment_names(2)[0]))
        self.assertTrue(
            np.all(
                [
                    x == 1
                    for i, x in enumerate(y_split_1)
                    if moments.LD.Util.moment_names(2)[0][i].split("_") == "DD"
                ]
            )
        )
        self.assertTrue(
            np.all(
                [
                    x == 2
                    for i, x in enumerate(y_split_1)
                    if moments.LD.Util.moment_names(2)[0][i].split("_") == "Dz"
                ]
            )
        )
        self.assertTrue(
            np.all(
                [
                    x == 3
                    for i, x in enumerate(y_split_1)
                    if moments.LD.Util.moment_names(2)[0][i].split("_") == "pi2"
                ]
            )
        )

    def test_split_pop_ids(self):
        pass


class SwapPops(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_swap_pops(self):
        rho = [0, 1, 2]
        y = moments.LD.Demographics1D.snm(rho=rho, theta=0.01)
        y = y.split(0)
        y = y.split(1)
        y.integrate([1, 2, 3], 0.01, theta=0.01, rho=rho)
        y_swap = y.swap_pops(0, 1)
        y_swap_back = y_swap.swap_pops(0, 1)
        for u, v in zip(y, y_swap_back):
            self.assertTrue(np.allclose(u, v))

        y.pop_ids = ["A", "B", "C"]
        y_swap = y.swap_pops(1, 2)
        self.assertTrue(
            np.all([x == y for x, y in zip(y_swap.pop_ids, ["A", "C", "B"])])
        )


class MarginalizeStats(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_marginalize(self):
        y = moments.LD.Demographics1D.snm()
        with self.assertRaises(ValueError):
            y.marginalize(0)
        with self.assertRaises(ValueError):
            y.marginalize([0])

        y = y.split(0)
        y = y.split(0)
        with self.assertRaises(ValueError):
            y.marginalize([0, 1, 2])
        y_marg = y.marginalize([0, 2])
        self.assertTrue(y_marg.num_pops == 1)

    def test_marginalize_pop_ids(self):
        y = moments.LD.Demographics1D.snm()
        y = y.split(0)
        y = y.split(0)
        y = y.split(0)
        y = y.split(0)
        y.pop_ids = ["a", "b", "c", "d", "e"]
        self.assertTrue(
            np.all(
                [x == y for x, y in zip(y.marginalize([0, 2]).pop_ids, ["b", "d", "e"])]
            )
        )


class SplitLD(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_split_1D(self):
        y = moments.LD.Demographics1D.snm(rho=[0, 10], pop_ids=["A"])
        y2 = y.split(0, new_ids=["B", "C"])
        for i, m in enumerate(y2.names()[0]):
            mm = m.split("_")[0]
            if mm == "DD":
                self.assertTrue(y2[0][i] == y[0][0])
            elif mm == "Dz":
                self.assertTrue(y2[0][i] == y[0][1])
            elif mm == "pi2":
                self.assertTrue(y2[0][i] == y[0][2])

    def test_split_2D(self):
        y = moments.LD.Demographics2D.split_mig(
            (2.0, 3.0, 0.1, 2.0), rho=1, pop_ids=["A", "B"]
        )
        y_s = y.split(1, new_ids=["C", "D"])
        self.assertTrue(y_s.pop_ids[0] == "A")
        self.assertTrue(y_s.pop_ids[1] == "C")
        self.assertTrue(y_s.pop_ids[2] == "D")
        self.assertTrue(
            y_s[0][y_s.names()[0].index("DD_1_2")] == y[0][y.names()[0].index("DD_1_1")]
        )
        self.assertTrue(
            y_s[0][y_s.names()[0].index("pi2_0_1_0_2")]
            == y[0][y.names()[0].index("pi2_0_1_0_1")]
        )

    def test_split_pop_ids(self):
        y = moments.LD.Demographics1D.snm(pop_ids=["a"])
        y = y.split(0, new_ids=["b", "c"])
        self.assertTrue(len(y.pop_ids) == 2)
        self.assertTrue(y.pop_ids[0] == "b")
        self.assertTrue(y.pop_ids[1] == "c")


class MergeLD(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_merge_two_pops(self):
        y = moments.LD.Demographics2D.split_mig(
            (1, 2, 0.1, 2), rho=1, pop_ids=["A", "B"]
        )
        with self.assertRaises(ValueError):
            y.merge(0, 1, 1.5)
        with self.assertRaises(ValueError):
            y.merge(0, 0, 0.5)
        with self.assertRaises(ValueError):
            y.merge(0, 2, 0.5)

        y1 = y.merge(0, 1, 0.5)
        self.assertTrue(y1.num_pops == 1)
        self.assertTrue(y1.pop_ids[0] == "Merged")

        y2 = y.merge(0, 1, 0.1, new_id="XX")
        self.assertTrue(y2.pop_ids[0] == "XX")


class AdmixLD(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_admix_two_pops(self):
        y = moments.LD.Demographics2D.split_mig(
            (1, 2, 0.1, 2), rho=1, pop_ids=["A", "B"]
        )
        with self.assertRaises(ValueError):
            y.admix(0, 1, 1.5)
        with self.assertRaises(ValueError):
            y.admix(0, 0, 0.5)
        with self.assertRaises(ValueError):
            y.admix(0, 2, 0.5)

        y1 = y.admix(0, 1, 0.5)
        self.assertTrue(y1.num_pops == 3)
        self.assertTrue(y1.pop_ids[0] == "A")
        self.assertTrue(y1.pop_ids[1] == "B")
        self.assertTrue(y1.pop_ids[2] == "Adm")

        y2 = y.admix(0, 1, 0.1, new_id="XX")
        self.assertTrue(y2.pop_ids[2] == "XX")

        y3 = y.merge(0, 1, 0.1, new_id="XX")
        y2 = y2.marginalize([0, 1])
        self.assertTrue(np.all(y3[0] == y2[0]))
        self.assertTrue(y3.pop_ids[0] == y2.pop_ids[0])


class PulseLD(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_pulse_two_pops(self):
        y = moments.LD.Demographics2D.split_mig(
            (1, 2, 0.1, 2), rho=1, pop_ids=["A", "B"]
        )
        with self.assertRaises(ValueError):
            y.pulse_migrate(0, 1, 1.5)
        with self.assertRaises(ValueError):
            y.pulse_migrate(0, 0, 0.5)
        with self.assertRaises(ValueError):
            y.pulse_migrate(0, 2, 0.5)

        y1 = y.pulse_migrate(0, 1, 0.1)
        self.assertTrue(y1.num_pops == 2)
        self.assertTrue(y1.pop_ids[0] == "A")
        self.assertTrue(y1.pop_ids[1] == "B")
        y2 = y.merge(0, 1, 0.1)
        y1 = y1.marginalize([0])
        self.assertTrue(np.all(y1[0] == y2[0]))


class TestDemographics1D(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def check_order(self, y_high, y_low):
        for m, x in zip(y_high.names()[0], y_high[0]):
            if m in y_low.names()[0]:
                self.assertTrue(x == y_low[0][y_low.names()[0].index(m)])

    def test_snm(self):
        y = moments.LD.Demographics1D.snm()
        self.assertEqual(len(y), 1)
        self.assertTrue(np.isclose(y[0][0], 0.001))

        y = moments.LD.Demographics1D.snm(pop_ids=["A"])
        self.assertTrue(y.pop_ids[0] == "A")

        y = moments.LD.Demographics1D.snm(rho=1.0)
        self.assertEqual(len(y), 2)

        y_0 = moments.LD.Demographics1D.snm(rho=0)
        y_1 = moments.LD.Demographics1D.snm(rho=1)
        y_0_1 = moments.LD.Demographics1D.snm(rho=[0, 1])
        self.assertTrue(np.allclose(y_0[0], y_0_1[0]))
        self.assertTrue(np.allclose(y_1[0], y_0_1[1]))

    def test_snm_order(self):
        rho = 1.5
        y2 = moments.LD.Demographics1D.snm(order=2, rho=rho)
        y4 = moments.LD.Demographics1D.snm(order=4, rho=rho)
        y6 = moments.LD.Demographics1D.snm(order=6, rho=rho)
        self.check_order(y6, y4)
        self.check_order(y4, y2)

    def test_two_epoch(self):
        y_snm = moments.LD.Demographics1D.snm(rho=1)
        y_2epoch = moments.LD.Demographics1D.two_epoch((1, 0.1), rho=1)
        self.assertTrue(np.allclose(y_snm[0], y_2epoch[0]))

        y_8 = moments.LD.Demographics1D.two_epoch(
            (2.0, 0.3), rho=2, theta=0.01, order=8, pop_ids=["XX"]
        )
        y_4 = moments.LD.Demographics1D.two_epoch(
            (2.0, 0.3), rho=2, theta=0.01, order=3, pop_ids=["XX"]
        )
        self.check_order(y_8, y_4)
        self.assertTrue(y_8.pop_ids[0] == y_4.pop_ids[0])

    def test_three_epoch(self):
        y_snm = moments.LD.Demographics1D.snm(rho=5, theta=0.05, pop_ids=["a"])
        y_3 = moments.LD.Demographics1D.three_epoch(
            (1, 1, 0.1, 0.1), rho=5, theta=0.05, pop_ids=["a"]
        )
        self.assertTrue(y_snm.pop_ids[0] == y_3.pop_ids[0])
        self.assertTrue(np.allclose(y_snm[0], y_3[0]))
        self.assertTrue(np.allclose(y_snm[1], y_3[1]))

        y_2 = moments.LD.Demographics1D.two_epoch((2, 0.1), rho=1)
        y_3a = moments.LD.Demographics1D.three_epoch((2, 2, 0.075, 0.025), rho=1)
        y_3b = moments.LD.Demographics1D.three_epoch((1, 2, 0.1, 0.1), rho=1)
        self.assertTrue(np.allclose(y_2[0], y_3a[0]))
        self.assertTrue(np.allclose(y_2[1], y_3a[1]))
        self.assertTrue(np.allclose(y_2[0], y_3b[0]))
        self.assertTrue(np.allclose(y_2[1], y_3b[1]))


class CopyAndPickle(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_copy(self):
        y = moments.LD.Demographics2D.snm(rho=[0, 1, 2], pop_ids=["A", "B"])
        y2 = copy.deepcopy(y)

    def test_pickle(self):
        y = moments.LD.Demographics2D.snm(rho=[0, 1, 2], pop_ids=["A", "B"])
        temp_file = "temp.bp"
        with open(temp_file, "wb+") as fout:
            pickle.dump(y, fout)
        with open(temp_file, "rb") as fin:
            y2 = pickle.load(fin)
        self.assertEqual(y.num_pops, y2.num_pops)
        self.assertEqual(y.pop_ids, y2.pop_ids)
        for x1, x2 in zip(y[:], y2[:]):
            self.assertTrue(np.all(x1 == x2))
        os.remove(temp_file)


class FStatistics(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_equivalenc(self):
        y = moments.LD.Demographics2D.snm()
        y = y.split(1)
        y.integrate([1, 2, 3], 0.2)
        self.assertTrue(y.f2(0, 1) == y.f3(0, 1, 1))
        self.assertTrue(y.f2(0, 1) == y.f2(1, 0))
        self.assertTrue(y.f2(0, 1) == y.f3(1, 0, 0))
        self.assertTrue(y.f2(0, 1) == y.f4(0, 1, 0, 1))
        self.assertTrue(y.f2(0, 1) == -y.f4(0, 1, 1, 0))


class DplusStatistic(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_steady_state(self):
        # at steady state, we expect it to be theta**2 as r -> 1/2 and
        # 2*theta**2 at r=0
        theta = 0.001
        y = moments.LD.Demographics1D.snm(theta=theta, rho=[0, 1000])
        H2 = y.H2(0)
        self.assertTrue(np.isclose(H2[0], 2 * theta ** 2))
        self.assertTrue(np.isclose(H2[1], theta ** 2))


# Older steady state functions
def steady_state(theta=0.001, rho=None, selfing_rate=None):
    if selfing_rate is None:
        h_ss = np.array([theta])
    else:
        h_ss = np.array([theta * (1 - selfing_rate / 2)])
    if hasattr(rho, "__len__"):  # list of rhos
        ys_ss = [
            equilibrium_ld(theta=theta, rho=r, selfing_rate=selfing_rate) for r in rho
        ]
        return ys_ss + [h_ss]
    elif np.isscalar(rho):  # one rho value
        y_ss = equilibrium_ld(theta=theta, rho=rho, selfing_rate=selfing_rate)
        return [y_ss, h_ss]
    else:  # only het stats
        return [h_ss]


def equilibrium_ld(theta=0.001, rho=0.0, selfing_rate=None):
    if selfing_rate is None:
        h_ss = np.array([theta])
    else:
        h_ss = np.array([theta * (1 - selfing_rate / 2)])
    U = Matrices.mutation_ld(1, theta, selfing=[selfing_rate])
    R = Matrices.recombination(1, rho, selfing=[selfing_rate])
    D = Matrices.drift_ld(1, [1.0])
    return factorized(D + R)(-U.dot(h_ss))


def steady_state_two_pop(nus, m, rho=None, theta=0.001, selfing_rate=None):
    nu0, nu1 = nus
    m01 = m[0][1]
    m10 = m[1][0]
    if selfing_rate is None:
        selfing_rate = [0, 0]

    # get the two-population steady state of heterozygosity statistics
    Mh = Matrices.migration_h(2, [[0, m01], [m10, 0]])
    Dh = Matrices.drift_h(2, [nu0, nu1])
    Uh = Matrices.mutation_h(2, theta, selfing=selfing_rate)
    h_ss = np.linalg.inv(Mh + Dh).dot(-Uh)

    # get the two-population steady state of LD statistics
    if rho is None:
        return [h_ss]

    def two_pop_ld_ss(nu0, nu1, m01, m10, theta, rho, selfing_rate, h_ss):
        U = Matrices.mutation_ld(2, theta, selfing=selfing_rate)
        R = Matrices.recombination(2, rho, selfing=selfing_rate)
        D = Matrices.drift_ld(2, [nu0, nu1])
        M = Matrices.migration_ld(2, [[0, m01], [m10, 0]])
        return factorized(D + R + M)(-U.dot(h_ss))

    if np.isscalar(rho):
        y_ss = two_pop_ld_ss(nu0, nu1, m01, m10, theta, rho, selfing_rate, h_ss)
        return [y_ss, h_ss]

    y_ss = []
    for r in rho:
        y_ss.append(two_pop_ld_ss(nu0, nu1, m01, m10, theta, r, selfing_rate, h_ss))
    return y_ss + [h_ss]


def steady_state_three_pop(nus, m, rho=None, theta=0.001, selfing_rate=None):
    nus = np.array(nus)
    m = np.array(m)
    if selfing_rate is None:
        selfing_rate = [0, 0, 0]

    # get the two-population steady state of heterozygosity statistics
    Mh = Matrices.migration_h(3, m)
    Dh = Matrices.drift_h(3, nus)
    Uh = Matrices.mutation_h(3, theta, selfing=selfing_rate)
    h_ss = np.linalg.inv(Mh + Dh).dot(-Uh)

    # get the two-population steady state of LD statistics
    if rho is None:
        return [h_ss]

    def three_pop_ld_ss(nus, m, theta, rho, selfing_rate, h_ss):
        U = Matrices.mutation_ld(3, theta, selfing=selfing_rate)
        R = Matrices.recombination(3, rho, selfing=selfing_rate)
        D = Matrices.drift_ld(3, nus)
        M = Matrices.migration_ld(3, m)
        return factorized(D + R + M)(-U.dot(h_ss))

    if np.isscalar(rho):
        y_ss = three_pop_ld_ss(nus, m, theta, rho, selfing_rate, h_ss)
        return [y_ss, h_ss]

    y_ss = []
    for r in rho:
        y_ss.append(three_pop_ld_ss(nus, m, theta, r, selfing_rate, h_ss))
    return y_ss + [h_ss]


class SteadyState(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_one_pop(self):
        for rho in [None, 0, 1, 10.5, [1, 2, 3]]:
            y = moments.LD.Demographics1D.snm(rho=rho)
            y.integrate([1], 1, rho=rho)
            y2 = moments.LD.LDstats(
                moments.LD.Numerics.steady_state([1], rho=rho), num_pops=1
            )
            for v, v2 in zip(y, y2):
                self.assertTrue(np.allclose(v, v2))

    def test_one_pop_selfing(self):
        for rho in [None, 0, 1, 10.5, [1, 2, 3]]:
            for f in [0, 0.25, 1]:
                y = moments.LD.LDstats(
                    moments.LD.Numerics.steady_state([1], rho=rho, selfing_rate=[f]),
                    num_pops=1,
                )
                y.integrate([1], 1, rho=rho, selfing=[f])
                y2 = moments.LD.LDstats(
                    moments.LD.Numerics.steady_state([1], rho=rho, selfing_rate=[f]),
                    num_pops=1,
                )
                for v, v2 in zip(y, y2):
                    self.assertTrue(np.all(np.allclose(v, v2)))

    def test_two_pops(self):
        nu0 = 1
        nu1 = 1
        m01 = 2.0
        m10 = 4.0
        nus = [nu0, nu1]
        m = [[0, m01], [m10, 0]]
        for selfing_rate in [None, [0.25, 0.5], [1, 1]]:
            for rho in [None, 2.5, [1, 3]]:
                if selfing_rate is None:
                    f0 = 0
                else:
                    f0 = np.mean(selfing_rate)
                y = moments.LD.LDstats(
                    moments.LD.Numerics.steady_state([1], rho=rho, selfing_rate=[f0]),
                    num_pops=1,
                )
                y = y.split(0)
                y.integrate(nus, 40, rho=rho, m=m, selfing=selfing_rate, dt_fac=0.02)
                y2 = moments.LD.LDstats(
                    moments.LD.Numerics.steady_state(
                        nus,
                        m,
                        rho=rho,
                        selfing_rate=selfing_rate,
                    ),
                    num_pops=2,
                )
                for v, v2 in zip(y, y2):
                    self.assertTrue(np.all(np.allclose(v, v2)))

    def test_steady_state_solution(self):
        y2 = moments.LD.LDstats(
            moments.LD.Numerics.steady_state([1, 1], [[0, 0], [1, 0]], rho=1),
            num_pops=2,
        )
        y1 = moments.LD.LDstats(
            moments.LD.Numerics.steady_state([1], rho=1), num_pops=1
        )
        for yy2, yy1 in zip(y2.marginalize(1), y1):
            self.assertTrue(np.allclose(yy2, yy1))
        y2 = moments.LD.LDstats(
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [0, 0]], rho=1),
            num_pops=2,
        )
        for yy2, yy1 in zip(y2.marginalize(0), y1):
            self.assertTrue(np.allclose(yy2, yy1))

    def test_bad_inputs_one_pop(self):
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1], theta=0)
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1], rho=-1)
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1], rho=[0, 1, -1])
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1], selfing_rate=[2])
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1], selfing_rate=[-1])

    def test_bad_inputs_two_pop(self):
        # theta cannot be zero
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [1, 0]], theta=0)
        # rho values must be non-negative
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [1, 0]], rho=-1)
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [1, 0]], rho=[0, 1, -1])
        # badly formed selfing rates
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [1, 0]], selfing_rate=0.5)
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [1, 0]], selfing_rate=[0])
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state(
                [1, 1], [[0, 1], [1, 0]], selfing_rate=[0, -1]
            )
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state(
                [1, 1], [[0, 1], [1, 0]], selfing_rate=[0, 2]
            )
        # bad pop sizes
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([-1, 1], [[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, -1], [[0, 1], [1, 0]])
        # bad migration rates
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, -1], [1, 0]])
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 1], [-1, 0]])
        with self.assertRaises(ValueError):
            moments.LD.Numerics.steady_state([1, 1], [[0, 0], [0, 0]])

    def test_connected_migration_matrix(self):
        M = [[0, 0], [0, 0]]
        self.assertFalse(moments.LD.Numerics._connected_migration_matrix(M))
        for M in [
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
            [[0, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]:
            self.assertTrue(moments.LD.Numerics._connected_migration_matrix(M))


class FromDemes(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_from_demes(self):
        b = demes.Builder()
        b.add_deme("A", epochs=[dict(start_size=1000, end_time=0)])
        g = b.resolve()
        y = moments.LD.LDstats.from_demes(g, sampled_demes=["A"], rho=0)
