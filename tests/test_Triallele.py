import os
import unittest

import numpy
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
        ns = 30
        cached = moments.Triallele.TriSpectrum.from_file('test_files/triallele_selection.fs')
        sA, hA = -1.0, 0.5
        sB, hB = -1.0, 0.5
        gammas = (2*sA,2*sA*hA,2*sB,2*sB*hB,2*sA*hA+2*sB*hB)
        fs = moments.Triallele.TriSpectrum(numpy.zeros((ns+1,ns+1)))
        fs.integrate(1.0, 20.0, gammas=gammas)
        self.assertTrue(numpy.allclose(fs, cached))
    
    def test_projection(self):
        ns1 = 20
        ns2 = 40
        xx1 = 1./numpy.linspace(0,ns1,ns1+1)
        xx1[0] = 1
        xx2 = 1./numpy.linspace(0,ns2,ns2+1)
        xx2[0] = 1
        fs1 = moments.Triallele.TriSpectrum(numpy.outer(xx1,xx1))
        fs2 = moments.Triallele.TriSpectrum(numpy.outer(xx2,xx2))
        fs_proj = fs2.project(ns1)
        self.assertTrue(numpy.allclose(fs_proj,fs1))

    def test_triallele_jackknife(self):
        ns1 = 40
        ns2 = 42
        xx1 = 1./numpy.linspace(0,ns1,ns1+1)
        xx1[0] = 1
        xx2 = 1./numpy.linspace(0,ns2,ns2+1)
        xx2[0] = 1
        fs1 = moments.Triallele.TriSpectrum(numpy.outer(xx1,xx1))
        fs2 = moments.Triallele.TriSpectrum(numpy.outer(xx2,xx2))
        J = moments.Triallele.Jackknife.calcJK_2(ns1)
        fs_jack = moments.Triallele.TriSpectrum(
                    moments.Triallele.Numerics.reform(
                        J.dot(moments.Triallele.Numerics.flatten(fs1)),ns2))
        self.assertTrue(numpy.allclose(fs_jack,fs2,rtol=0.04))

suite = unittest.TestLoader().loadTestsFromTestCase(TrialleleTestCase)

if __name__ == '__main__':
    unittest.main()
