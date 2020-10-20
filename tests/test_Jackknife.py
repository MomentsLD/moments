import unittest
import numpy
import moments
import time


class JackknifeTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_jk_one_jump(self):
        n = 100
        xx = [1.0 / i for i in range(1, n)]
        yy = [1.0 / i for i in range(1, n + 1)]
        J = moments.Jackknife.calcJK13(n)
        zz = numpy.dot(J, xx)
        self.assertTrue(numpy.allclose(zz, yy, atol=3e-04))

    def test_jk_two_jumps(self):
        n = 100
        xx = [1.0 / i for i in range(1, n)]
        yy = [1.0 / i for i in range(1, n + 2)]
        J = moments.Jackknife.calcJK23(n)
        zz = numpy.dot(J, xx)
        self.assertTrue(numpy.allclose(zz, yy, atol=8e-04))


suite = unittest.TestLoader().loadTestsFromTestCase(JackknifeTestCase)
if __name__ == "__main__":
    unittest.main()
