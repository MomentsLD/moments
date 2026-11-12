import os, unittest
import contextlib
import copy
import io
import tempfile
import numpy as np
import moments, moments.TwoLocus
import time


class TestTwoLocusMethods(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()
        # files are written to a temporary directory, so that tests running in
        # parallel (pytest -n) don't overwrite each other's files
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_to_from_file(self):
        # make some data
        comments = ["comment 1", "comment 2"]
        filename = os.path.join(self.tmpdir.name, "test.fs")
        data = np.random.rand(11, 11, 11)
        fs = moments.TwoLocus.TLSpectrum(data)
        # test saving and reloading data fs
        fs.to_file(filename, comment_lines=comments)
        fs_from = moments.TwoLocus.TLSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(np.allclose(fs, fs_from))

        # test saving with folding attribute
        fs_folded = fs.fold()
        fs_folded.to_file(filename)
        fs_from = moments.TwoLocus.TLSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(np.allclose(fs_folded, fs_from))

    def test_phi_to_array(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "test_files/two_locus_ns{0}_rho0.fs".format(ns),
            )
        )
        out = moments.TwoLocus.Numerics.Phi_to_array(
            moments.TwoLocus.Numerics.array_to_Phi(cached), ns
        )
        self.assertTrue(np.all(cached.data == out))

    def test_neutral(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "test_files/two_locus_ns{0}_rho0.fs".format(ns),
            )
        )
        fs = moments.TwoLocus.TLSpectrum(np.zeros((ns + 1, ns + 1, ns + 1)))
        fs.integrate(1, 20, rho=0.0, dt=0.2)
        self.assertTrue(np.allclose(fs, cached))

    # def test_recombination(self):
    #    ns = 30
    #    cached = moments.TwoLocus.TLSpectrum.from_file(
    #        os.path.join(
    #            os.path.dirname(__file__),
    #            "test_files/two_locus_ns{0}_rho1.fs".format(ns),
    #        )
    #    )
    #    fs = moments.TwoLocus.TLSpectrum(np.zeros((ns + 1, ns + 1, ns + 1)))
    #    fs.integrate(1, 20, rho=1.0, dt=0.2)
    #    self.assertTrue(np.allclose(fs, cached))

    def test_jackknife_project(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "test_files/two_locus_ns{0}_rho1.fs".format(ns),
            )
        )
        J1 = moments.TwoLocus.Jackknife.calc_jk(ns, 1)
        J2 = moments.TwoLocus.Jackknife.calc_jk(ns, 2)
        F1 = moments.TwoLocus.TLSpectrum(
            moments.TwoLocus.Numerics.Phi_to_array(
                J1.dot(moments.TwoLocus.Numerics.array_to_Phi(cached)), ns + 1
            )
        )
        F2 = moments.TwoLocus.TLSpectrum(
            moments.TwoLocus.Numerics.Phi_to_array(
                J2.dot(moments.TwoLocus.Numerics.array_to_Phi(cached)), ns + 2
            )
        )
        out1 = F1.project(ns)
        out2 = F2.project(ns)
        out1.mask_fixed()
        out2.mask_fixed()
        cached.mask_fixed()
        self.assertTrue(np.allclose(out2, cached, atol=0.02, rtol=0.01))
        self.assertTrue(np.allclose(out1, cached, atol=0.02, rtol=0.01))


class TestTwoLocusResults(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    # def test_equilibrium(self):
    #    ns = 30
    #    rhos = [0, 1]
    #    for rho in rhos:
    #        cached = moments.TwoLocus.TLSpectrum.from_file(
    #            os.path.join(
    #                os.path.dirname(__file__),
    #                f"test_files/two_locus_ns{ns}_rho{rho}.fs",
    #            )
    #        )
    #        F = moments.TwoLocus.Demographics.equilibrium(ns, rho=rho)
    #        self.assertTrue(np.allclose(F, cached))

    def test_additive_general_selection(self):
        gamma = -2
        ns = 30
        rhos = [0, 1, 10]
        for rho in rhos:
            F1 = moments.TwoLocus.Demographics.equilibrium(
                ns, rho=rho, sel_params=[2 * gamma, gamma, gamma]
            )
            F2 = moments.TwoLocus.Demographics.equilibrium(
                ns,
                rho=rho,
                sel_params_general=[
                    4 * gamma,
                    3 * gamma,
                    3 * gamma,
                    2 * gamma,
                    2 * gamma,
                    2 * gamma,
                    gamma,
                    2 * gamma,
                    gamma,
                ],
            )
            self.assertTrue(np.allclose(F1.data, F2.data, atol=0.001))

    def test_integration_neutral(self):
        ns = 30
        rhos = [0, 1]
        for rho in rhos:
            F0 = moments.TwoLocus.Demographics.equilibrium(ns, rho=rho)
            F1 = copy.deepcopy(F0)
            F1.integrate(1.0, 0.1, rho=rho)
            self.assertTrue(np.allclose(F0.data, F1.data, atol=0.0005))

    def test_integration_sel_params_general(self):
        ns = 30
        rho = 1
        s = 1
        sel_params = [2 * s, s, s]
        sel_params_general = [4 * s, 3 * s, 3 * s, 2 * s, 2 * s, 2 * s, s, 2 * s, s]

        F1 = moments.TwoLocus.Demographics.equilibrium(
            ns, rho=rho, sel_params=sel_params
        )
        F2 = moments.TwoLocus.Demographics.equilibrium(
            ns, rho=rho, sel_params_general=sel_params_general
        )

        self.assertTrue(np.allclose(F1.data, F2.data, atol=1e-6, rtol=1e-2))
        self.assertTrue(
            np.isclose(F1.D() / F1.pi2(), F2.D() / F2.pi2(), atol=1e-4, rtol=1e-2)
        )
        self.assertTrue(
            np.isclose(F1.D2() / F1.pi2(), F2.D2() / F2.pi2(), atol=1e-4, rtol=1e-2)
        )

        F1.integrate(2, 0.05, rho=rho, sel_params=sel_params)
        F2.integrate(2, 0.05, rho=rho, sel_params_general=sel_params_general)

        self.assertTrue(np.allclose(F1.data, F2.data, atol=1e-5, rtol=1e-2))
        self.assertTrue(
            np.isclose(F1.D() / F1.pi2(), F2.D() / F2.pi2(), atol=1e-4, rtol=1e-2)
        )
        self.assertTrue(
            np.isclose(F1.D2() / F1.pi2(), F2.D2() / F2.pi2(), atol=1e-4, rtol=1e-2)
        )


class TestEquilibriumCache(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()
        self.old_cache_path = moments.TwoLocus.Demographics.cache_path
        self.tmpdir = tempfile.TemporaryDirectory()
        moments.TwoLocus.Demographics.set_cache_path(self.tmpdir.name)

    def tearDown(self):
        moments.TwoLocus.Demographics.cache_path = self.old_cache_path
        self.tmpdir.cleanup()
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_cache_distinguishes_sel_params_general(self):
        # Regression test for #294: spectra cached with sel_params_general used to be
        # saved under the neutral file name, so later neutral calls (or calls with other
        # general selection parameters) loaded the wrong spectrum. Cache three different
        # models and check that each cached load matches a fresh computation.
        n = 14
        rho = 1.0
        eq = moments.TwoLocus.Demographics.equilibrium
        sel_1 = moments.TwoLocus.Util.simple_dominance(-2.0, h=0.1)
        sel_2 = moments.TwoLocus.Util.gene_based_dominance(-2.0, h=0.1)
        F_sel_1 = eq(n, rho=rho, sel_params_general=sel_1, cache=True)
        F_sel_2 = eq(n, rho=rho, sel_params_general=sel_2, cache=True)
        F_neu = eq(n, rho=rho, cache=True)
        self.assertEqual(len(os.listdir(self.tmpdir.name)), 3)
        self.assertFalse(np.allclose(F_sel_1.data, F_neu.data))
        self.assertFalse(np.allclose(F_sel_1.data, F_sel_2.data))
        # reloading from the cache gives back the matching spectrum
        for sel, F in [(sel_1, F_sel_1), (sel_2, F_sel_2), (None, F_neu)]:
            F_cached = eq(n, rho=rho, sel_params_general=sel, cache=True)
            F_fresh = eq(n, rho=rho, sel_params_general=sel, cache=False)
            self.assertTrue(np.allclose(F_cached.data, F.data))
            self.assertTrue(np.allclose(F_cached.data, F_fresh.data))


class TestSelectionArguments(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_gamma_is_additive_at_both_loci(self):
        # Checks the documented meaning of gamma (#293): it is shorthand for
        # sel_params = [2 * gamma, gamma, gamma], i.e. additive selection at both
        # loci, so that the single-locus spectrum stored along F[0, :, 0] (left locus)
        # and F[0, 0, :] (right locus) matches the single-site SFS with the same gamma.
        n = 10
        gamma = -2.0
        F = moments.TwoLocus.Demographics.equilibrium(n, rho=0, gamma=gamma)
        F_sel = moments.TwoLocus.Demographics.equilibrium(
            n, rho=0, sel_params=[2 * gamma, gamma, gamma]
        )
        self.assertTrue(np.allclose(F.data, F_sel.data))
        fs = moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma, h=0.5)
        self.assertTrue(np.allclose(F[0, 1:-1, 0], fs[1:-1]))
        self.assertTrue(np.allclose(F[0, 0, 1:-1], fs[1:-1]))

    def test_conflicting_selection_arguments_raise(self):
        # Giving more than one selection model used to either silently ignore gamma
        # or fail with an unrelated error, so each combination should now raise a
        # ValueError, in both integrate and equilibrium.
        n = 14
        sel = [0.2, 0.1, 0.1]
        sel_gen = moments.TwoLocus.Util.simple_dominance(-1.0, h=0.1)
        F = moments.TwoLocus.Demographics.equilibrium(n, rho=1.0)
        conflicts = [
            dict(gamma=-1.0, sel_params=sel),
            dict(gamma=-1.0, sel_params_general=sel_gen),
            dict(sel_params=sel, sel_params_general=sel_gen),
        ]
        for kwargs in conflicts:
            with self.assertRaises(ValueError):
                F.copy().integrate(1.0, 0.01, rho=1.0, **kwargs)
            with self.assertRaises(ValueError):
                moments.TwoLocus.Demographics.equilibrium(n, rho=1.0, **kwargs)
        # only gamma (at the left locus) is supported with the finite genome model
        for kwargs in [dict(sel_params=sel), dict(sel_params_general=sel_gen)]:
            with self.assertRaises(ValueError):
                F.copy().integrate(
                    1.0, 0.01, rho=1.0, finite_genome=True, u=1e-3, v=1e-3, **kwargs
                )

    def test_neutral_models_do_not_print(self):
        # Demographic models used to print a message about gamma even when no
        # selection was specified. Capture stdout for a neutral model with rho given.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            moments.TwoLocus.Demographics.two_epoch([2.0, 0.1], 10, rho=1.0)
        self.assertEqual(buf.getvalue(), "")

    def test_small_sample_size_jackknife_error(self):
        # The jackknife is not defined for small sample sizes; check that we raise a
        # ValueError with a useful message instead of an IndexError.
        with self.assertRaises(ValueError):
            moments.TwoLocus.Jackknife.calc_jk(6, 1)
        with self.assertRaises(ValueError):
            moments.TwoLocus.Jackknife.calc_jk(12, 2)


class TestLowOrderStats(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_signed_D_conditioned(self):
        n = 10
        for nA, nB in np.random.randint(1, n // 2 + 1, size=(10, 2)):
            F = moments.TwoLocus.TLSpectrum(np.zeros((n + 1, n + 1, n + 1)))
            for i in range(min(nA, nB) + 1):
                F[i, nA - i, nB - i] = np.random.rand()

            D_1 = F.D()
            D_2 = F.D(nA=nA, nB=nB)
            self.assertTrue(np.isclose(D_1, D_2))
            D_1 = F.D(proj=False)
            D_2 = F.D(proj=False, nA=nA, nB=nB)
            self.assertTrue(np.isclose(D_1, D_2))
            D_1 = moments.TwoLocus.Util.compute_D_threshold(F, thresh=5)
            D_2 = moments.TwoLocus.Util.compute_D_conditional(F, nAmax=5, nBmax=5)
            self.assertTrue(np.isclose(D_1, D_2))

    def test_D2_conditioned(self):
        n = 10
        for nA, nB in np.random.randint(1, n // 2 + 1, size=(10, 2)):
            F = moments.TwoLocus.TLSpectrum(np.zeros((n + 1, n + 1, n + 1)))
            for i in range(min(nA, nB) + 1):
                F[i, nA - i, nB - i] = np.random.rand()

            D2_1 = F.D2()
            D2_2 = F.D2(nA=nA, nB=nB)
            self.assertTrue(np.isclose(D2_1, D2_2))
            D2_1 = F.D2(proj=False)
            D2_2 = F.D2(proj=False, nA=nA, nB=nB)
            self.assertTrue(np.isclose(D2_1, D2_2))
            D2_1 = moments.TwoLocus.Util.compute_D2_threshold(F, thresh=5)
            D2_2 = moments.TwoLocus.Util.compute_D2_conditional(F, nAmax=5, nBmax=5)
            self.assertTrue(np.isclose(D2_1, D2_2))

    def test_Dz_conditioned(self):
        n = 10
        for nA, nB in np.random.randint(1, n // 2 + 1, size=(10, 2)):
            F = moments.TwoLocus.TLSpectrum(np.zeros((n + 1, n + 1, n + 1)))
            for i in range(min(nA, nB) + 1):
                F[i, nA - i, nB - i] = np.random.rand()

            Dz_1 = F.Dz()
            Dz_2 = F.Dz(nA=nA, nB=nB)
            self.assertTrue(np.isclose(Dz_1, Dz_2))
            Dz_1 = F.Dz(proj=False)
            Dz_2 = F.Dz(proj=False, nA=nA, nB=nB)
            self.assertTrue(np.isclose(Dz_1, Dz_2))
            Dz_1 = moments.TwoLocus.Util.compute_Dz_threshold(F, thresh=5)
            Dz_2 = moments.TwoLocus.Util.compute_Dz_conditional(F, nAmax=5, nBmax=5)
            self.assertTrue(np.isclose(Dz_1, Dz_2))

    def test_pi2_conditioned(self):
        n = 10
        for nA, nB in np.random.randint(1, n // 2 + 1, size=(10, 2)):
            F = moments.TwoLocus.TLSpectrum(np.zeros((n + 1, n + 1, n + 1)))
            for i in range(min(nA, nB) + 1):
                F[i, nA - i, nB - i] = np.random.rand()

            pi2_1 = F.pi2()
            pi2_2 = F.pi2(nA=nA, nB=nB)
            self.assertTrue(np.isclose(pi2_1, pi2_2))
            pi2_1 = F.pi2(proj=False)
            pi2_2 = F.pi2(proj=False, nA=nA, nB=nB)
            self.assertTrue(np.isclose(pi2_1, pi2_2))
            pi2_1 = moments.TwoLocus.Util.compute_pi2_threshold(F, thresh=5)
            pi2_2 = moments.TwoLocus.Util.compute_pi2_conditional(F, nAmax=5, nBmax=5)
            self.assertTrue(np.isclose(pi2_1, pi2_2))

    def test_S(self):
        # TLSpectrum.S used to raise an AttributeError. Compare it to a direct sum over
        # entries where both loci are segregating, overall and for given nA and nB.
        n = 8
        rng = np.random.default_rng(42)
        F = moments.TwoLocus.TLSpectrum(rng.random((n + 1, n + 1, n + 1)))
        total = 0
        total_nA = 0
        total_nA_nB = 0
        for i in range(n + 1):
            for j in range(n + 1 - i):
                for k in range(n + 1 - i - j):
                    if i + j in [0, n] or i + k in [0, n]:
                        continue
                    total += F.data[i, j, k]
                    if i + j == 3:
                        total_nA += F.data[i, j, k]
                        if i + k == 2:
                            total_nA_nB += F.data[i, j, k]
        self.assertTrue(np.isclose(F.S(), total))
        self.assertTrue(np.isclose(F.S(nA=3), total_nA))
        self.assertTrue(np.isclose(F.S(nA=3, nB=2), total_nA_nB))


if __name__ == "__main__":
    unittest.main()
