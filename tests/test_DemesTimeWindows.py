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


class TestSNM(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_single_window(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Deme", epochs=[dict(start_size=10000, end_time=0)])
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])

        fss = Demes.SFS(
            g, sampled_demes=["Deme"], sample_sizes=[20], mutation_time_windows=[0]
        )

        self.assertTrue(np.all(fs.data == fss[0].data))
        self.assertTrue(fs.pop_ids == fss[0].pop_ids)

    def test_two_windows(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Deme", epochs=[dict(start_size=10000, end_time=0)])
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            mutation_time_windows=[0, 1000],
        )

        self.assertTrue(np.allclose(fs, fss[0] + fss[1]))
        self.assertTrue(fs.pop_ids == fss[0].pop_ids)
        self.assertTrue(fss[0].pop_ids == fss[1].pop_ids)

    def test_long_window(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Deme", epochs=[dict(start_size=10000, end_time=0)])
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            mutation_time_windows=[0, 100 * 10000],
        )

        self.assertTrue(np.allclose(fs, fss[0] + fss[1]))
        self.assertTrue(np.allclose(fs, fss[0]))
        self.assertTrue(np.allclose(fss[1], 0))
        self.assertTrue(fs.pop_ids == fss[0].pop_ids)

    def test_many_window(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Deme", epochs=[dict(start_size=10000, end_time=0)])
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            mutation_time_windows=[
                0,
                100,
                300,
                1000,
                1200,
                1201,
                1500,
                10000,
                20000,
                21000,
                21002,
            ],
        )

        self.assertTrue(fss[0].pop_ids == fs.pop_ids)
        for fs2 in fss[1:]:
            self.assertTrue(fss[0].pop_ids == fs2.pop_ids)

        fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
        self.assertTrue(np.allclose(fs, fs_comb))

    def test_window_order(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Deme", epochs=[dict(start_size=10000, end_time=0)])
        g = b.resolve()

        n = 100
        fs = moments.Spectrum(np.zeros(n + 1))
        fs.integrate([1], 1, theta=4 * 10000)

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[n],
            mutation_time_windows=[0, 20000],
        )

        fss2 = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[n],
            mutation_time_windows=[0, 2000, 20000],
        )

        fss3 = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[n],
            mutation_time_windows=[0, 20000, 40000],
        )

        self.assertTrue(np.allclose(fs, fss[0]))
        self.assertTrue(np.allclose(fs, fss2[0] + fss2[1], atol=1e-2, rtol=1e-5))
        self.assertTrue(np.allclose(fs, fss3[0]))


class TestBadInputs(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_time_windows(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme("Deme", epochs=[dict(start_size=10000, end_time=0)])
        g = b.resolve()

        with self.assertRaises(ValueError):
            fs = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=[-1, 0, 100],
            )
        with self.assertRaises(ValueError):
            fs = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=[0, 100, 100, 500],
            )
        with self.assertRaises(ValueError):
            fs = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=[10, 100],
            )
        with self.assertRaises(ValueError):
            fs = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=[0, 0, 100],
            )
        with self.assertRaises(ValueError):
            fs = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=[],
            )


class TestPiecewiseConstant(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_nonoverlapping_windows(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Deme",
            epochs=[
                dict(start_size=10000, end_time=1000),
                dict(start_size=5000, end_time=500),
                dict(start_size=20000, end_time=0),
            ],
        )
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])

        for tw in [[0, 20000], [0, 10, 100, 7500], [0, 2000, 20000]]:
            fss = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=tw,
            )
            fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
            self.assertTrue(np.allclose(fs, fs_comb))

    def test_overlapping_windows(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Deme",
            epochs=[
                dict(start_size=10000, end_time=1000),
                dict(start_size=5000, end_time=500),
                dict(start_size=20000, end_time=0),
            ],
        )
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])

        for tw in [
            [0, 1000],
            [0, 500],
            [0, 500, 1000],
            [0, 500, 2000],
            [0, 500, 1000, 4000],
        ]:
            fss = Demes.SFS(
                g,
                sampled_demes=["Deme"],
                sample_sizes=[20],
                mutation_time_windows=tw,
            )
            fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
            self.assertTrue(np.allclose(fs, fs_comb))


class TestTwoPopulation(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_nonoverlapping_windows(self):
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Anc",
            epochs=[
                dict(start_size=10000, end_time=1000),
            ],
        )
        b.add_deme("Deme1", ancestors=["Anc"], epochs=[dict(start_size=20000)])
        b.add_deme("Deme2", ancestors=["Anc"], epochs=[dict(start_size=5000)])
        g = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme1", "Deme2"], sample_sizes=[20, 30])

        for tw in [[0, 20000], [0, 1000, 7500], [0, 2000, 20000]]:
            fss = Demes.SFS(
                g,
                sampled_demes=["Deme1", "Deme2"],
                sample_sizes=[20, 30],
                mutation_time_windows=tw,
            )
            fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
            self.assertTrue(np.allclose(fs, fs_comb))


class TestAncientSamples(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_snm(self):
        Ne = 1e4
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Deme",
            epochs=[
                dict(start_size=Ne),
            ],
        )
        g = b.resolve()

        fs = Demes.SFS(
            g,
            sampled_demes=["Deme", "Deme"],
            sample_sizes=[20, 20],
            sample_times=[0, 500],
        )

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme", "Deme"],
            sample_sizes=[20, 20],
            sample_times=[0, 500],
            mutation_time_windows=[0, 500, 1000, 2000],
        )

        fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
        self.assertTrue(np.allclose(fs, fs_comb))

        fs_comb1 = moments.Spectrum(np.sum(fss[1:], axis=0), pop_ids=fss[0].pop_ids)
        self.assertTrue(np.allclose(fs.marginalize([0]), fs_comb.marginalize([0])))

        self.assertTrue(np.all(fss[0].marginalize([0]) == 0))
        self.assertTrue(np.all(fss[0].marginalize([1]) > 0))
        self.assertTrue(np.all(fss[1].marginalize([0]) > 0))
        self.assertTrue(np.all(fss[1].marginalize([1]) > 0))

    def test_all_ancient_samples(self):
        Ne = 1e4
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Deme",
            epochs=[
                dict(start_size=Ne),
            ],
        )
        g = b.resolve()

        fs = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            sample_times=[500],
        )

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            sample_times=[500],
            mutation_time_windows=[0, 500, 1000, 2000],
        )

        fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
        self.assertTrue(np.allclose(fs, fs_comb))

        self.assertTrue(len(fss) == 4)
        self.assertTrue(np.all(fss[0] == 0))

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            sample_times=[500],
            mutation_time_windows=[0, 100, 200, 500, 1000, 2000],
        )

        fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
        self.assertTrue(np.allclose(fs, fs_comb))

        self.assertTrue(len(fss) == 6)
        self.assertTrue(np.all(fss[0] == 0))
        self.assertTrue(np.all(fss[1] == 0))
        self.assertTrue(np.all(fss[2] == 0))

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            sample_times=[500],
            mutation_time_windows=[0, 100, 200, 400, 1000, 2000],
        )

        fs_comb = moments.Spectrum(np.sum(fss, axis=0), pop_ids=fss[0].pop_ids)
        self.assertTrue(np.allclose(fs, fs_comb))

        self.assertTrue(len(fss) == 6)
        self.assertTrue(np.all(fss[0] == 0))
        self.assertTrue(np.all(fss[1] == 0))


class TestTimeUnits(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_snm_years(self):
        Ne = 1e4
        gen = 25
        b = demes.Builder(description="test", time_units="generations")
        b.add_deme(
            "Deme",
            epochs=[
                dict(start_size=Ne),
            ],
        )
        g = b.resolve()

        b = demes.Builder(description="test", time_units="years", generation_time=gen)
        b.add_deme(
            "Deme",
            epochs=[
                dict(start_size=Ne),
            ],
        )
        g_years = b.resolve()

        fs = Demes.SFS(g, sampled_demes=["Deme"], sample_sizes=[20])
        fs_years = Demes.SFS(g_years, sampled_demes=["Deme"], sample_sizes=[20])
        self.assertTrue(np.all(fs == fs_years))

        fss = Demes.SFS(
            g,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            mutation_time_windows=[0, 500, 1000, 2000],
        )

        fss_years = Demes.SFS(
            g_years,
            sampled_demes=["Deme"],
            sample_sizes=[20],
            mutation_time_windows=[_ * gen for _ in [0, 500, 1000, 2000]],
        )

        for fs1, fs2 in zip(fss, fss_years):
            self.assertTrue(np.all(fs1 == fs2))
