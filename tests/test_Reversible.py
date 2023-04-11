import os
import unittest

import numpy as np
import scipy.special
import moments

# these are slow because integrating to equilibrium takes a while
# since we are starting with entries only in fixed bins
import time


class FiniteGenomeTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_reversible_neutral(self):
        ns = 30
        theta_fd = 1e-3
        theta_bd = 3e-3
        exact = moments.LinearSystem_1D.steady_state_1D_reversible(
            ns, theta_fd=theta_fd, theta_bd=theta_bd
        )
        fs = moments.Spectrum(exact)
        fs.unmask_all()
        # integrate for T = 2 and check that it's close still
        fs.integrate([1.0], 2, finite_genome=True, theta_fd=theta_fd, theta_bd=theta_bd)
        self.assertTrue(np.allclose(fs, exact))

    def test_reversible_selection(self):
        gammas = [1, -1, -10]
        theta_fd = 2e-3
        theta_bd = 1e-3
        for gamma in gammas:
            ns = 30
            exact = moments.LinearSystem_1D.steady_state_1D_reversible(
                ns, gamma=gamma, theta_fd=theta_fd, theta_bd=theta_bd
            )
            # integrate for T=1 and check that result stayed close
            fs = moments.Spectrum(exact)
            fs.unmask_all()
            fs.integrate(
                [1.0],
                1,
                finite_genome=True,
                gamma=gamma,
                theta_fd=theta_fd,
                theta_bd=theta_bd,
            )
            self.assertTrue(np.allclose(fs, exact, atol=1e-5))

    def test_two_pop(self):
        ns1 = 30
        ns2 = 20
        ns = ns1 + ns2
        theta_fd = 2e-3
        theta_bd = 1e-3
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                ns, theta_fd=theta_fd, theta_bd=theta_bd
            ),
            mask_corners=False,
        )
        fs2 = moments.Manips.split_1D_to_2D(fs, 30, 20)
        fs2.integrate(
            [1.0, 1.0], 1, finite_genome=True, theta_fd=theta_fd, theta_bd=theta_bd
        )
        fsm1 = fs2.marginalize([1], mask_corners=False)
        fsm2 = fs2.marginalize([0], mask_corners=False)
        exact1 = moments.LinearSystem_1D.steady_state_1D_reversible(
            ns1, theta_fd=theta_fd, theta_bd=theta_bd
        )
        exact2 = moments.LinearSystem_1D.steady_state_1D_reversible(
            ns2, theta_fd=theta_fd, theta_bd=theta_bd
        )
        self.assertTrue(
            np.allclose(fsm1, exact1, atol=1e-5)
            and np.allclose(fsm2, exact2, atol=1e-5)
        )

    def test_two_pop_with_migration(self):
        n1 = 30
        n2 = 30
        n = n1 + n2
        u = 1e-3
        v = 2e-3
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                n, theta_fd=u, theta_bd=v
            ),
            mask_corners=False,
        )
        fs = fs.split(0, n1, n2)
        m = 1
        fs.integrate(
            [1, 1], 2, m=[[0, m], [0, 0]], theta_fd=u, theta_bd=v, finite_genome=True
        )
        fs_marg = fs.marginalize([0])
        fs1 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                n2, theta_fd=u, theta_bd=v
            ),
            mask_corners=False,
        )
        self.assertTrue(np.allclose(fs1, fs_marg, rtol=0.001))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                n, theta_fd=u, theta_bd=v
            ),
            mask_corners=False,
        )
        fs2 = fs2.split(0, n1, n2)
        fs2.integrate(
            [1, 1], 2, m=[[0, 0], [m, 0]], theta_fd=u, theta_bd=v, finite_genome=True
        )
        fs2_marg = fs2.marginalize([1])
        self.assertTrue(np.allclose(fs2_marg, fs_marg))

    def test_frozen(self):
        n = 20
        u = 1e-3
        v = 2e-3
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                n, theta_fd=u, theta_bd=v
            ),
            mask_corners=False,
        )
        fs_ss = fs.copy()
        fs.integrate([10], 1, theta_fd=u, theta_bd=v, finite_genome=True, frozen=[True])
        self.assertTrue(np.all(fs_ss == fs))
        fs.integrate([10], 1, theta_fd=u, theta_bd=v, finite_genome=True)
        self.assertFalse(np.allclose(fs_ss, fs))
        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                2 * n, theta_fd=u, theta_bd=v
            ),
            mask_corners=False,
        )
        fs2 = fs2.split(0, n, n)
        self.assertTrue(np.allclose(fs_ss, fs2.marginalize([1])))
        self.assertTrue(np.allclose(fs_ss, fs2.marginalize([0])))
        fs2.integrate(
            [10, 10],
            1,
            theta_fd=u,
            theta_bd=v,
            finite_genome=True,
            frozen=[False, True],
        )
        self.assertFalse(np.allclose(fs_ss, fs2.marginalize([1])))
        self.assertTrue(np.allclose(fs_ss, fs2.marginalize([0])))

    def test_three_pop(self):
        n = 10
        u = 1e-3
        v = 2e-3
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                3 * n, theta_fd=u, theta_bd=v
            ),
            mask_corners=False,
        )
        fs_ss = fs.project([10])
        fs = fs.split(0, 20, 10)
        fs = fs.split(0, 10, 10)
        fs.integrate([1, 1, 1], 1, theta_fd=u, theta_bd=v, finite_genome=True)
        self.assertTrue(np.allclose(fs_ss, fs.marginalize([0, 1]), rtol=1e-4))
        self.assertTrue(np.allclose(fs_ss, fs.marginalize([0, 2]), rtol=1e-4))
        self.assertTrue(np.allclose(fs_ss, fs.marginalize([1, 2]), rtol=1e-4))

        n = 30
        u = 1e-3
        v = 2e-3
        gamma = -2
        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D_reversible(
                3 * n, theta_fd=u, theta_bd=v, gamma=gamma
            ),
            mask_corners=False,
        )
        fs_ss = fs.project([n])
        fs = fs.split(0, 2 * n, n)
        fs = fs.split(0, n, n)
        fs.integrate(
            [1, 1, 1], 1, theta_fd=u, theta_bd=v, finite_genome=True, gamma=gamma
        )
        self.assertTrue(np.allclose(fs_ss, fs.marginalize([0, 1]), rtol=2e-3))
        self.assertTrue(np.allclose(fs_ss, fs.marginalize([0, 2]), rtol=2e-3))
        self.assertTrue(np.allclose(fs_ss, fs.marginalize([1, 2]), rtol=2e-3))


suite = unittest.TestLoader().loadTestsFromTestCase(FiniteGenomeTestCase)

if __name__ == "__main__":
    unittest.main()
