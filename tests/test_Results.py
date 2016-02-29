import os
import unittest
import numpy
import moments

class ResultsTestCase(unittest.TestCase):
    def test_1d_ic(self):
        # This just the standard neutral model
        n = 10
        fs = moments.Spectrum(numpy.zeros(n+1))
        fs.integrate([1], [n], tf=10, dt_fac=0.01)
        answer = moments.Spectrum(1./numpy.arange(n+1))
        self.assert_(numpy.ma.allclose(fs, answer, atol=5e-5))

    def test_1pop(self):
        n = 15
        f = lambda x: [1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n+1]))
        sfs.integrate(f, [n], 5, 0.01, theta=1.0, h=[0.1], gamma=[-1], m=[0])
        sfs_ref = moments.Spectrum.from_file('test_files/1_pop.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))
    
    def test_2pops_neutral(self):
        n = 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n+1, n+1]))
        sfs.integrate(f, [n, n], 10, 0.005, theta=1.0, h=[0.5, 0.5], gamma=[0, 0], m=mig)
        sfs_ref = moments.Spectrum.from_file('test_files/2_pops_neutral.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))
    
    def test_2pops(self):
        n1, n2 = 15, 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n1+1, n2+1]))
        sfs.integrate(f, [n1, n2], 10, 0.005, theta=1.0, h=[0.6, 0.6], gamma=[2, 2], m=mig)
        sfs_ref = moments.Spectrum.from_file('test_files/2_pops.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))
    
    def test_3pops(self):
        n1, n2, n3 = 15, 20, 18
        n2 = 20
        n3 = 18
        gamma = [0, 0.5, -2]
        h = [0.5, 0.1, 0.9]
        mig = numpy.array([[0, 5, 2],[1, 0, 1],[10, 0, 1]])
        f = lambda x: [1, 1, 1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n1+1, n2+1, n3+1]))
        sfs.integrate(f, [n1, n2, n3], 10, 0.005, theta=1.0, h=h, gamma=gamma, m=mig)
        sfs_ref = moments.Spectrum.from_file('test_files/3_pops.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

    def test_IM(self):
        params = (0.8, 2.0, 0.6, 0.45, 5.0, 0.3)
        ns = (7,13)
        pts_l = 50
        theta = 1000.
        fs = theta*moments.Demographics2D.IM(params, ns)

        msfs = moments.Spectrum.from_file('test_files/IM.fs')

        resid = moments.Inference.Anscombe_Poisson_residual(fs,msfs)
        
        self.assert_(abs(resid).max() < 0.25)

suite = unittest.TestLoader().loadTestsFromTestCase(ResultsTestCase)

if __name__ == '__main__':
    unittest.main()
