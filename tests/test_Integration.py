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
        M = moments.Integration.calcM([25.0, 30.0], m)
        self.assertTrue(numpy.allclose(M[0].todense(), Mref))

suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestCase)
if __name__ == '__main__':
    unittest.main()
