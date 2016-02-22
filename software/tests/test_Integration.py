import unittest
import numpy
import moments

class IntegrationTestCase(unittest.TestCase):
    def test_matrix_drift(self):
        Dref = dlim = numpy.genfromtxt('test_files/drift_matrix.csv', delimiter=',')
        D = moments.Integration.calcD([25, 30])
        d = D[0][0].todense() + D[0][1].todense()
        self.assertTrue(numpy.allclose(d, Dref))
    
    def test_matrix_selection_1(self):
        S1ref = dlim = numpy.genfromtxt('test_files/selection_matrix_1.csv', delimiter=',')
        S1 = moments.Integration.calcS([25, 30], [-1, 1], [0.1, 0.4])
        self.assertTrue(numpy.allclose(S1[0].todense(), S1ref))
    
    def test_matrix_selection_2(self):
        S2ref = dlim = numpy.genfromtxt('test_files/selection_matrix_2.csv', delimiter=',')
        S2 = moments.Integration.calcS2([25, 30], [-1, 1], [0.1, 0.4])
        self.assertTrue(numpy.allclose(S2[0].todense(), S2ref))

    def test_matrix_migration(self):
        Mref = dlim = numpy.genfromtxt('test_files/migration_matrix.csv', delimiter=',')
        m = numpy.array([[1, 5],[10, 1]])
        M = moments.Integration.calcM([25, 30], m)
        self.assertTrue(numpy.allclose(M[0].todense(), Mref))

    def test_1pop(self):
        n = 15
        f = lambda x: [1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n+1]))
        sfs.integrate(f, [n], 5, 0.05, theta=1.0, h=[0.1], gamma=[-1], m=[0])
        sfs_ref = moments.Spectrum.from_file('test_files/1_pop.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))
    
    def test_2pops_neutral(self):
        n = 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n+1, n+1]))
        sfs.integrate(f, [n, n], 10, 0.05, theta=1.0, h=[0.5, 0.5], gamma=[0, 0], m=mig)
        sfs_ref = moments.Spectrum.from_file('test_files/2_pops_neutral.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))
    
    def test_2pops(self):
        n1, n2 = 15, 20
        mig = numpy.ones([2, 2])
        f = lambda x: [1, 1+0.0001*x]
        sfs = moments.Spectrum(numpy.zeros([n1+1, n2+1]))
        sfs.integrate(f, [n1, n2], 10, 0.05, theta=1.0, h=[0.6, 0.6], gamma=[2, 2], m=mig)
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
        sfs.integrate(f, [n1, n2, n3], 10, 0.05, theta=1.0, h=h, gamma=gamma, m=mig)
        sfs_ref = moments.Spectrum.from_file('test_files/3_pops.fs')
        self.assertTrue(numpy.allclose(sfs, sfs_ref))

suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestCase)
if __name__ == '__main__':
    unittest.main()
