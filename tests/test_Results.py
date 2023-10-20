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
        sfs.integrate(f, 5, dt_fac=0.01, theta=1.0, h=0.1, gamma=-1)
        sfs_ref = moments.Spectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/1_pop.fs")
        )
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

    def test_2pops_neutral(self):
        n = 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1 + 0.0001 * x]
        sfs = moments.Spectrum(numpy.zeros([n + 1, n + 1]))
        sfs.integrate(f, 10, dt_fac=0.005, theta=1.0, h=[0.5, 0.5], gamma=[0, 0], m=mig)
        sfs_ref = moments.Spectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/2_pops_neutral.fs")
        )
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

    def test_2pops(self):
        n1, n2 = 15, 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1 + 0.0001 * x]
        sfs = moments.Spectrum(numpy.zeros([n1 + 1, n2 + 1]))
        sfs.integrate(f, 10, dt_fac=0.005, theta=1.0, h=[0.6, 0.6], gamma=[2, 2], m=mig)
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


class IntegrationValidityTestCase(unittest.TestCase):
    def test_negative_integration_time(self):
        fs = moments.Demographics1D.snm([20])
        with self.assertRaises(ValueError):
            fs.integrate([1], -1)

    def test_negative_integration_sizes(self):
        fs = moments.Demographics1D.snm([20])
        with self.assertRaises(ValueError):
            fs.integrate([-1], 1)
        with self.assertRaises(ValueError):
            fs.integrate([0], 1)
        with self.assertRaises(ValueError):
            nu_func = lambda t: [-1 + 2 * t]
            fs.integrate(nu_func, 1)
        fs = moments.Demographics2D.snm([20, 20])
        with self.assertRaises(ValueError):
            fs.integrate([0, 1], 1)
        with self.assertRaises(ValueError):
            fs.integrate([1, -1], 1)
        with self.assertRaises(ValueError):
            nu_func = lambda t: [0, 1]
            fs.integrate(nu_func, 1)
        with self.assertRaises(ValueError):
            nu_func = lambda t: [1, -1]
            fs.integrate(nu_func, 1)


class TimeDependentSelection(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_single_population_constant_selection(self):
        n = 20
        gamma = -2
        # given as single value
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))
        fs.integrate([1], 0.3, gamma=gamma)

        # given as an array
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))
        fs2.integrate([1], 0.3, gamma=[gamma])
        self.assertTrue(np.allclose(fs, fs2))

        # given as a function
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))
        gamma_func = lambda t: gamma
        fs2.integrate([1], 0.3, gamma=gamma_func)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))
        gamma_func = lambda t: [gamma]
        fs2.integrate([1], 0.3, gamma=gamma_func)
        self.assertTrue(np.allclose(fs, fs2))

        # h also given, then as a function
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))
        gamma_func = lambda t: gamma
        h_func = lambda t: 0.5
        fs2.integrate([1], 0.3, gamma=gamma_func)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma))
        gamma_func = lambda t: gamma
        h_func = lambda t: [0.5]
        fs2.integrate([1], 0.3, gamma=gamma_func)
        self.assertTrue(np.allclose(fs, fs2))

    def test_single_population_pw_constant_selection(self):
        n = 20
        gamma1 = -2
        T1 = 0.2
        nu1 = 2
        gamma2 = -5
        T2 = 0.3
        nu2 = 0.5
        # given as single value
        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n))
        fs.integrate([nu1], T1, gamma=gamma1, dt_fac=0.005)
        fs.integrate([nu2], T2, gamma=gamma2, dt_fac=0.005)

        gamma_func = lambda t: gamma1 if t < T1 else gamma2
        nu_func = lambda t: [nu1] if t < T1 else [nu2]
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n))
        fs2.integrate(nu_func, T1 + T2, gamma=gamma_func, dt_fac=0.005)

        self.assertTrue(np.allclose(fs, fs2))

        # vary h as well
        h1 = 0.2
        h2 = 0.8
        h_func = lambda t: h1 if t < T1 else h2

        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n))
        fs.integrate([nu1], T1, gamma=gamma1, h=h1, dt_fac=0.005)
        fs.integrate([nu2], T2, gamma=gamma2, h=h2, dt_fac=0.005)
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n))
        fs2.integrate(nu_func, T1 + T2, gamma=gamma_func, h=h_func, dt_fac=0.005)

        self.assertTrue(np.allclose(fs, fs2))

    def test_single_population_nonconstant(self):
        n = 30
        T = 0.5
        gamma0 = -5
        gamma_func = lambda t: gamma0 * np.exp(-t / T)

        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(n, gamma=gamma0))
        fs.integrate([1], T, gamma=gamma_func)

    def test_two_pop_nomig_constant_equal(self):
        n = 20
        T = 0.3
        gamma = -3

        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], T, gamma=gamma)

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: gamma
        fs2.integrate([1, 1], T, gamma=gamma_func)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma, gamma]
        fs2.integrate([1, 1], T, gamma=gamma_func)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma, gamma]
        h_func = lambda t: 0.5
        fs2.integrate([1, 1], T, gamma=gamma_func, h=h_func)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma, gamma]
        h_func = lambda t: [0.5, 0.5]
        fs2.integrate([1, 1], T, gamma=gamma_func, h=h_func)
        self.assertTrue(np.allclose(fs, fs2))

    def test_two_pop_nomig_constant_unequal(self):
        n = 20
        T = 0.3
        gamma0 = -3
        gamma1 = 2
        gamma2 = 0.1
        h = 0.1
        h1 = 0.4
        h2 = 1.1

        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h)
        )
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], T, gamma=[gamma1, gamma2], h=[h1, h2])

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma1, gamma2]
        fs2.integrate([1, 1], T, gamma=gamma_func, h=[h1, h2])
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma1, gamma2]
        h_func = lambda t: [h1, h2]
        fs2.integrate([1, 1], T, gamma=gamma_func, h=h_func)
        self.assertTrue(np.allclose(fs, fs2))

    def test_two_pop_nomig_pwconstant(self):
        n = 20

        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=1))
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], 0.1, gamma=[-1, 2], dt_fac=0.005)
        fs.integrate([1, 1], 0.1, gamma=[-2, 2], dt_fac=0.005)
        fs.integrate([1, 1], 0.1, gamma=[-2, 1], dt_fac=0.005)

        gamma_func = lambda t: [-1, 2] if t < 0.1 else [-2, 2] if t < 0.2 else [-2, 1]
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=1))
        fs2 = fs2.split(0, n, n)
        fs2.integrate([1, 1], 0.3, gamma=gamma_func, dt_fac=0.005)

        self.assertTrue(np.allclose(fs, fs2, rtol=0.002))

    def test_two_pop_mig_constant_equal(self):
        n = 20
        T = 0.3
        gamma = -3
        m = [[0, 2], [1, 0]]

        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], T, gamma=gamma, m=m)

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: gamma
        fs2.integrate([1, 1], T, gamma=gamma_func, m=m)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma, gamma]
        fs2.integrate([1, 1], T, gamma=gamma_func, m=m)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma, gamma]
        h_func = lambda t: 0.5
        fs2.integrate([1, 1], T, gamma=gamma_func, h=h_func, m=m)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma, gamma]
        h_func = lambda t: [0.5, 0.5]
        fs2.integrate([1, 1], T, gamma=gamma_func, h=h_func, m=m)
        self.assertTrue(np.allclose(fs, fs2))

    def test_two_pop_mig_constant_unequal(self):
        n = 20
        T = 0.3
        gamma0 = -3
        gamma1 = 2
        gamma2 = 0.1
        h = 0.1
        h1 = 0.4
        h2 = 1.1
        m = [[0, 2], [1, 0]]

        fs = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h)
        )
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], T, gamma=[gamma1, gamma2], h=[h1, h2], m=m)

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma1, gamma2]
        fs2.integrate([1, 1], T, gamma=gamma_func, h=[h1, h2], m=m)
        self.assertTrue(np.allclose(fs, fs2))

        fs2 = moments.Spectrum(
            moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=gamma0, h=h)
        )
        fs2 = fs2.split(0, n, n)
        gamma_func = lambda t: [gamma1, gamma2]
        h_func = lambda t: [h1, h2]
        fs2.integrate([1, 1], T, gamma=gamma_func, h=h_func, m=m)
        self.assertTrue(np.allclose(fs, fs2))

    def test_two_pop_mig_pwconstant(self):
        n = 20
        m = [[0, 2], [1, 0]]

        fs = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=1))
        fs = fs.split(0, n, n)
        fs.integrate([1, 1], 0.1, gamma=[-1, 2], m=m, dt_fac=0.005)
        fs.integrate([1, 1], 0.1, gamma=[-2, 2], m=m, dt_fac=0.005)
        fs.integrate([1, 1], 0.1, gamma=[-2, 1], m=m, dt_fac=0.005)

        gamma_func = lambda t: [-1, 2] if t < 0.1 else [-2, 2] if t < 0.2 else [-2, 1]
        fs2 = moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(2 * n, gamma=1))
        fs2 = fs2.split(0, n, n)
        fs2.integrate([1, 1], 0.3, gamma=gamma_func, m=m, dt_fac=0.005)

        self.assertTrue(np.allclose(fs, fs2, rtol=0.002))


suite = unittest.TestLoader().loadTestsFromTestCase(ResultsTestCase)

if __name__ == "__main__":
    unittest.main()
