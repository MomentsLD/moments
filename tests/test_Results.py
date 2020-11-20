import os
import unittest
import numpy, numpy as np
import moments
import time


class ResultsTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_1d_ic(self):
        # This just the standard neutral model
        n = 10
        fs = moments.Spectrum(numpy.zeros(n + 1))
        fs.integrate([1], tf=10, dt_fac=0.01)
        answer = moments.Spectrum(1.0 / numpy.arange(n + 1))
        self.assertTrue(numpy.ma.allclose(fs, answer, atol=5e-5))

    def test_1pop(self):
        n = 15
        f = lambda x: [1 + 0.0001 * x]
        sfs = moments.Spectrum(numpy.zeros([n + 1]))
        sfs.integrate(f, 5, 0.01, theta=1.0, h=0.1, gamma=-1)
        sfs_ref = moments.Spectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/1_pop.fs")
        )
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

    def test_2pops_neutral(self):
        n = 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1 + 0.0001 * x]
        sfs = moments.Spectrum(numpy.zeros([n + 1, n + 1]))
        sfs.integrate(f, 10, 0.005, theta=1.0, h=[0.5, 0.5], gamma=[0, 0], m=mig)
        sfs_ref = moments.Spectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/2_pops_neutral.fs")
        )
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

    def test_2pops(self):
        n1, n2 = 15, 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1 + 0.0001 * x]
        sfs = moments.Spectrum(numpy.zeros([n1 + 1, n2 + 1]))
        sfs.integrate(f, 10, 0.005, theta=1.0, h=[0.6, 0.6], gamma=[2, 2], m=mig)
        sfs_ref = moments.Spectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/2_pops.fs")
        )
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

    def test_IM(self):
        params = (0.8, 2.0, 0.6, 0.45, 5.0, 0.3)
        ns = (20, 13)
        theta = 1000.0
        fs = theta * moments.Demographics2D.IM(params, ns)

        dadi_fs = moments.Spectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/IM.fs")
        )

        resid = moments.Inference.Anscombe_Poisson_residual(fs, dadi_fs)

        self.assertTrue(abs(resid).max() < 0.25)

    def test_selection_specification(self):
        gamma = -2
        h = 0.2
        fs = moments.Demographics2D.snm([15, 15])
        fs.integrate([2, 3], 0.3, gamma=[gamma, gamma], h=[h, h])
        fs2 = moments.Demographics2D.snm([15, 15])
        fs2.integrate([2, 3], 0.3, gamma=gamma, h=h)
        self.assertTrue(np.allclose(fs.data, fs2.data))

suite = unittest.TestLoader().loadTestsFromTestCase(ResultsTestCase)

if __name__ == "__main__":
    unittest.main()
