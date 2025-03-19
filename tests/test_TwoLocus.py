import os, unittest
import copy
import numpy as np
import moments, moments.TwoLocus
import time


class TestTwoLocusMethods(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_to_from_file(self):
        # make some data
        comments = ["comment 1", "comment 2"]
        filename = "test.fs"
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

    #def test_recombination(self):
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

    #def test_equilibrium(self):
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


suite = unittest.TestLoader().loadTestsFromTestCase(TestTwoLocusMethods)
suite = unittest.TestLoader().loadTestsFromTestCase(TestTwoLocusResults)
suite = unittest.TestLoader().loadTestsFromTestCase(TestLowOrderStats)

if __name__ == "__main__":
    unittest.main()
