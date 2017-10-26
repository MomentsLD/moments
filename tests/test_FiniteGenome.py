import os
import unittest

import numpy
import scipy.special
import moments

# these are slow because integrating to equilibrium takes a while 
# since we are starting with entries only in fixed bins

class FiniteGenomeTestCase(unittest.TestCase):
    def test_reversible_neutral(self):
        ns = 30
        theta_fd = 2e-3
        theta_bd = 1e-3
        exact = moments.LinearSystem_1D.steady_state_1D_reversible(ns, 
                                            theta_fd=theta_fd, theta_bd=theta_bd)
        fs = moments.Spectrum(numpy.zeros(ns+1), mask_corners=False)
        fs[0] = exact[0]
        fs[-1] = exact[-1]
        fs /= numpy.sum(fs)
        fs.integrate([1.0], 5000, finite_genome=True, 
                                            theta_fd=theta_fd, theta_bd=theta_bd)
        self.assertTrue(numpy.allclose(fs, exact))
    
    def test_reversible_selection(self):
        gamma = -5.0
        theta_fd = 2e-3
        theta_bd = 1e-3
        ns = 30
        exact = moments.LinearSystem_1D.steady_state_1D_reversible(ns, gamma=gamma,
                                            theta_fd=theta_fd, theta_bd=theta_bd)
        fs = moments.Spectrum(numpy.zeros(ns+1), mask_corners=False)
        fs[0] = exact[0]
        fs[-1] = exact[-1]
        fs /= numpy.sum(fs)
        fs.integrate([1.0], 5000, finite_genome=True, gamma=gamma,
                                            theta_fd=theta_fd, theta_bd=theta_bd)
        self.assertTrue(numpy.allclose(fs, exact, atol=1e-5))
        

suite = unittest.TestLoader().loadTestsFromTestCase(FiniteGenomeTestCase)

if __name__ == '__main__':
    unittest.main()
