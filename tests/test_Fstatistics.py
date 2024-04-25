import os
import unittest
import numpy as np
import moments
import time
import demes


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
