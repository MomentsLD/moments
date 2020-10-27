import os
import unittest

import numpy
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
        self.assertTrue(numpy.allclose(fs, exact))

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
            self.assertTrue(numpy.allclose(fs, exact, atol=1e-5))

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
            numpy.allclose(fsm1, exact1, atol=1e-5)
            and numpy.allclose(fsm2, exact2, atol=1e-5)
        )


suite = unittest.TestLoader().loadTestsFromTestCase(FiniteGenomeTestCase)

if __name__ == "__main__":
    unittest.main()
