import os
import unittest

import numpy
import scipy.special
import moments, moments.Triallele

# these are slow because integrating to equilibrium takes a while 
# since we are starting with entries only in fixed bins

class TrialleleTestCase(unittest.TestCase):
    def test_neutral(self):
        ns = 30
        xx = 1./numpy.linspace(0,ns,ns+1)
        xx[0] = 1.
        exact = moments.Triallele.TriSpectrum(numpy.outer(xx,xx))
        fs = moments.Triallele.TriSpectrum(numpy.zeros((ns+1,ns+1)))
        fs.integrate(1.0, 20.0)
        self.assertTrue(numpy.allclose(fs, exact))
    
    def test_selection(self):
        pass
#        ns = 30
#        cached = moments.Triallele.TriSpectrum.from_file('test_files/triallele_selection.fs')
#        sA, hA = -1.0, 0.5
#        sB, hB = -1.0, 0.5
#        gammas = (2*sA,2*sA*hA,2*sB,2*sB*hB,2*sA*hA+2*sB*hB)
#        fs = moments.Triallele.TriSpectrum(np.zeros((ns+1,ns+1)))
#        fs.integrate(1.0, 20.0, gammas=gammas)
#        self.assertTrue(numpy.allclose(fs, exact))
    
    def test_projection(self):
        pass
    
    def test_triallele_jackknife(self):
        pass

suite = unittest.TestLoader().loadTestsFromTestCase(TrialleleTestCase)

if __name__ == '__main__':
    unittest.main()
