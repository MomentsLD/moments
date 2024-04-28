import os
import unittest
import numpy as np
import moments
import time
import demes
import copy


class FStatistics(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_LD_equivalence(self):
        y = moments.LD.Demographics2D.snm()
        y = y.split(1)
        y.integrate([1, 2, 3], 0.2)
        self.assertTrue(y.f2(0, 1) == y.f3(0, 1, 1))
        self.assertTrue(y.f2(0, 1) == y.f2(1, 0))
        self.assertTrue(y.f2(0, 1) == y.f3(1, 0, 0))
        self.assertTrue(y.f2(0, 1) == y.f4(0, 1, 0, 1))
        self.assertTrue(y.f2(0, 1) == -y.f4(0, 1, 1, 0))

    def test_F2_LD_against_SFS(self):
        b = demes.Builder()
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=200)])
        b.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=100, end_size=5000)])
        b.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=2000, end_size=500)])
        b.add_migration(demes=["A", "B"], rate=1e-4)
        g = b.resolve()

        y = moments.Demes.LD(g, ["A", "B"], u=1e-6)
        fs = moments.Demes.SFS(g, samples={"A": 20, "B": 20}, u=1e-6)

        f2_LD = y.f2("A", "B")
        f2_SFS = fs.f2("A", "B")

        self.assertTrue(np.isclose(f2_LD, f2_SFS, rtol=2e-4))

    def test_F3_LD_against_SFS(self):
        b = demes.Builder()
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=200)])
        b.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=2000)])
        b.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=3000)])
        b.add_deme("C", ancestors=["anc"], epochs=[dict(start_size=4000)])
        g = b.resolve()
        y = moments.Demes.LD(g, ["A", "B", "C"], u=1e-6)
        fs = moments.Demes.SFS(g, samples={"A": 20, "B": 20, "C": 20}, u=1e-6)

        assert np.isclose(y.f3(0, 1, 2), fs.f3(0, 1, 2))
        assert np.isclose(y.f3(1, 0, 2), fs.f3(1, 0, 2))
        assert np.isclose(y.f3(2, 1, 0), fs.f3(2, 1, 0))
        assert np.isclose(y.f3(0, 2, 1), fs.f3(0, 2, 1))
        assert np.isclose(y.f3(1, 2, 0), fs.f3(1, 2, 0))
        assert np.isclose(y.f3(2, 0, 1), fs.f3(2, 0, 1))

    def test_ordering(self):
        y = moments.LD.Demographics2D.snm()
        y = y.split(1)
        y.integrate([1, 2, 3], 0.1)
        y = y.split(0)
        y.integrate([1, 2, 3, 4], 0.1)

        assert np.isclose(y.f2(0, 1), y.f2(1, 0))
        assert np.isclose(y.f2(0, 3), y.f2(3, 0))
        assert np.isclose(y.f4(0, 1, 2, 3), y.f4(1, 0, 3, 2))
        assert np.isclose(y.f4(0, 1, 2, 3), -y.f4(1, 0, 2, 3))

        y2 = copy.deepcopy(y)
        y2 = y2.swap_pops(0, 3)
        assert np.isclose(y.f3(0, 2, 3), y2.f3(3, 2, 0))
        assert np.isclose(y.f3(0, 1, 2), y.f3(0, 2, 1))

    def test_ordering_SFS(self):
        fs = moments.Demographics2D.snm([12, 12])
        fs = fs.split(1, 6, 6)
        fs.integrate([1, 2, 3], 0.1)
        fs = fs.split(0, 6, 6)
        fs.integrate([1, 2, 3, 4], 0.1)

        assert np.isclose(fs.f2(0, 1), fs.f2(1, 0))
        assert np.isclose(fs.f2(0, 3), fs.f2(3, 0))
        assert np.isclose(fs.f4(0, 1, 2, 3), fs.f4(1, 0, 3, 2))
        assert np.isclose(fs.f4(0, 1, 2, 3), -fs.f4(1, 0, 2, 3))

        fs2 = copy.deepcopy(fs)
        fs2 = fs2.swap_axes(0, 3)
        assert np.isclose(fs.f3(0, 2, 3), fs2.f3(3, 2, 0))
        assert np.isclose(fs.f3(0, 1, 2), fs.f3(0, 2, 1))

    def test_SFS_equiv_index_names(self):
        size = (4, 5, 6, 7, 8)
        fs = moments.Spectrum(np.random.rand(np.prod(size)).reshape(size))
        fs.pop_ids = ["A", "B", "C", "D", "E"]

        for i in range(5):
            x = fs.pop_ids[i]
            for j in range(5):
                y = fs.pop_ids[j]
                assert fs.f2(i, j) == fs.f2(x, y)
                for k in range(5):
                    z = fs.pop_ids[k]
                    assert fs.f3(i, j, k) == fs.f3(x, y, z)
                    for l in range(5):
                        w = fs.pop_ids[l]
                        assert fs.f4(i, j, k, l) == fs.f4(x, y, z, w)

    def test_ordering_equiv_models(self):
        b = demes.Builder()
        b.add_deme("anc", epochs=[dict(start_size=1000, end_time=200)])
        b.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=2000)])
        b.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=3000)])
        b.add_deme("C", ancestors=["anc"], epochs=[dict(start_size=4000)])
        b.add_deme("D", ancestors=["anc"], epochs=[dict(start_size=5000)])
        g = b.resolve()

        b2 = demes.Builder()
        b2.add_deme("anc", epochs=[dict(start_size=1000, end_time=200)])
        b2.add_deme("D", ancestors=["anc"], epochs=[dict(start_size=5000)])
        b2.add_deme("C", ancestors=["anc"], epochs=[dict(start_size=4000)])
        b2.add_deme("B", ancestors=["anc"], epochs=[dict(start_size=3000)])
        b2.add_deme("A", ancestors=["anc"], epochs=[dict(start_size=2000)])
        g2 = b2.resolve()

        y = moments.Demes.LD(g, ["A", "B", "C", "D"], u=1e-6)
        y2 = moments.Demes.LD(g2, ["D", "C", "B", "A"], u=1e-6)

        assert y.f2("A", "C") == y2.f2("A", "C")
        assert y.f3("B", "A", "C") == y2.f3("B", "C", "A")

        fs = moments.Demes.SFS(g, samples={"A": 10, "B": 10, "C": 10, "D": 10}, u=1e-6)
        fs2 = moments.Demes.SFS(
            g2, samples={"D": 10, "C": 10, "B": 10, "A": 10}, u=1e-6
        )

        assert np.isclose(fs.f2("A", "C"), fs2.f2("A", "C"))
        assert np.isclose(fs.f3("B", "A", "C"), fs2.f3("B", "C", "A"))

        assert np.isclose(fs.f2("B", "C"), y.f2("B", "C"))
        assert np.isclose(fs.f3("C", "A", "B"), y.f3("C", "A", "B"))

        assert np.isclose(y.f4("A", "B", "C", "D"), y2.f4("A", "B", "C", "D"))
        assert np.isclose(fs.f4("A", "B", "C", "D"), fs2.f4("A", "B", "C", "D"))
