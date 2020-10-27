import os, unittest

import numpy
import moments, moments.TwoLocus
import time


class TwoLocusTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_to_from_file(self):
        # make some data
        comments = ["comment 1", "comment 2"]
        filename = "test.fs"
        data = numpy.random.rand(11, 11, 11)
        fs = moments.TwoLocus.TLSpectrum(data)
        # test saving and reloading data fs
        fs.to_file(filename, comment_lines=comments)
        fs_from = moments.TwoLocus.TLSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs, fs_from))

        # test saving with folding attribute
        fs_folded = fs.fold()
        fs_folded.to_file(filename)
        fs_from = moments.TwoLocus.TLSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs_folded, fs_from))

    def test_neutral(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "test_files/two_locus_ns{0}_rho0.fs".format(ns),
            )
        )
        fs = moments.TwoLocus.TLSpectrum(numpy.zeros((ns + 1, ns + 1, ns + 1)))
        fs.integrate(1, 20, rho=0.0, dt=0.2)
        self.assertTrue(numpy.allclose(fs.data, cached.data))

    def test_recombination(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "test_files/two_locus_ns{0}_rho1.fs".format(ns),
            )
        )
        fs = moments.TwoLocus.TLSpectrum(numpy.zeros((ns + 1, ns + 1, ns + 1)))
        fs.integrate(1, 20, rho=1.0, dt=0.2)
        self.assertTrue(numpy.allclose(fs, cached))


suite = unittest.TestLoader().loadTestsFromTestCase(TwoLocusTestCase)

if __name__ == "__main__":
    unittest.main()
