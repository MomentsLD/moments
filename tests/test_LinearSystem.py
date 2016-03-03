import unittest
import numpy
import moments

import sys
sys.path[:0] = ['../moments/']

import Jackknife as jk
import LinearSystem_2D

class LinearSystemTestCase(unittest.TestCase):
    def test_matrix_drift_2D(self):
        Dref = dlim = numpy.genfromtxt('test_files/drift_matrix.csv', delimiter=',')
        d1 = LinearSystem_2D.calcD1([25, 30])
        d2 = LinearSystem_2D.calcD2([25, 30])
        d = d1.todense() + d2.todense()
        self.assertTrue(numpy.allclose(d, Dref))

    def test_matrix_selection_1_2D(self):
        S1ref = dlim = numpy.genfromtxt('test_files/selection_matrix_1.csv', delimiter=',')
        dims=[25, 30]
        ljk = [jk.calcJK13(int(dims[i]-1)) for i in range(len(dims))]
        s1 = LinearSystem_2D.calcS_1(dims, ljk[0])
        s2 = LinearSystem_2D.calcS_2(dims, ljk[1])
        S1 = -0.1*s1+0.4*s2
        self.assertTrue(numpy.allclose(S1.todense(), S1ref))

    def test_matrix_selection_2_SD(self):
        S2ref = dlim = numpy.genfromtxt('test_files/selection_matrix_2.csv', delimiter=',')
        dims=[25, 30]
        ljk = [jk.calcJK23(int(dims[i]-1)) for i in range(len(dims))]
        s1 = LinearSystem_2D.calcS2_1(dims, ljk[0])
        s2 = LinearSystem_2D.calcS2_2(dims, ljk[1])
        S2 = -0.8*s1+0.2*s2
        self.assertTrue(numpy.allclose(S2.todense(), S2ref))

    def test_matrix_migration(self):
        Mref = dlim = numpy.genfromtxt('test_files/migration_matrix.csv', delimiter=',')
        dims=[25, 30]
        m = numpy.array([[1, 5],[10, 1]])
        ljk = [jk.calcJK13(int(dims[i]-1)) for i in range(len(dims))]
        #M1 = moments.Integration.calcM([25, 30], m)[0].todense()
        m1 = LinearSystem_2D.calcM_1(dims, ljk[1])
        m2 = LinearSystem_2D.calcM_2(dims, ljk[0])
        M = m[0,1]*m1+m[1,0]*m2
        '''M2 = M.todense()
        for i in range(25):
            for j in range(30):
              print(i, j, M1[i,j], M2[i,j])'''
        self.assertTrue(numpy.allclose(M.todense(), Mref))


suite = unittest.TestLoader().loadTestsFromTestCase(LinearSystemTestCase)
if __name__ == '__main__':
    unittest.main()
