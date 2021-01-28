import os, unittest

import numpy as np
import moments, moments.TwoLocus
import time


class TwoLocusMethods(unittest.TestCase):
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
        self.assertTrue(np.allclose(fs.data, cached.data))

    def test_recombination(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "test_files/two_locus_ns{0}_rho1.fs".format(ns),
            )
        )
        fs = moments.TwoLocus.TLSpectrum(np.zeros((ns + 1, ns + 1, ns + 1)))
        fs.integrate(1, 20, rho=1.0, dt=0.2)
        self.assertTrue(np.allclose(fs, cached))

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
        self.assertTrue(np.allclose(out2, cached, atol=0.01, rtol=0.01))
        self.assertTrue(np.allclose(out1, cached, atol=0.01, rtol=0.01))


class TwoLocusResults(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_equilibrium(self):
        ns = 30
        rhos = [0, 1]
        for rho in rhos:
            cached = moments.TwoLocus.TLSpectrum.from_file(
                os.path.join(
                    os.path.dirname(__file__),
                    f"test_files/two_locus_ns{ns}_rho{rho}.fs",
                )
            )
            F = moments.TwoLocus.Demographics.equilibrium(ns, rho=rho)
            self.assertTrue(np.allclose(F, cached))

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
            self.assertTrue(np.allclose(F1.data, F2.data, atol=0.0005))

suite = unittest.TestLoader().loadTestsFromTestCase(TwoLocusMethods)
suite = unittest.TestLoader().loadTestsFromTestCase(TwoLocusResults)

if __name__ == "__main__":
    unittest.main()
