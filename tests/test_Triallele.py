import os
import unittest
import copy
import numpy
import moments, moments.Triallele
import time


class TrialleleTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_to_from_file(self):
        # make some data
        comments = ["comment 1", "comment 2"]
        filename = "test.fs"
        data = numpy.random.rand(10, 10)
        fs = moments.Triallele.TriSpectrum(data)
        # test saving and reloading data fs
        fs.to_file(filename, comment_lines=comments)
        fs_from = moments.Triallele.TriSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs, fs_from))

        # test saving with folding attributes
        fs_anc = fs.fold_ancestral()
        fs_anc.to_file(filename)
        fs_from = moments.Triallele.TriSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs_anc, fs_from) and fs_from.folded_ancestral)

        fs_maj = fs.fold_major()
        fs_maj.to_file(filename)
        fs_from = moments.Triallele.TriSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs_maj, fs_from) and fs_from.folded_major)

    def test_projection(self):
        ns1 = 20
        ns2 = 40
        xx1 = 1.0 / numpy.linspace(0, ns1, ns1 + 1)
        xx1[0] = 1
        xx2 = 1.0 / numpy.linspace(0, ns2, ns2 + 1)
        xx2[0] = 1
        fs1 = moments.Triallele.TriSpectrum(numpy.outer(xx1, xx1))
        fs2 = moments.Triallele.TriSpectrum(numpy.outer(xx2, xx2))
        fs_proj = fs2.project(ns1)
        self.assertTrue(numpy.allclose(fs_proj, fs1))

    def test_triallele_jackknife(self):
        ns1 = 40
        ns2 = 42
        xx1 = 1.0 / numpy.linspace(0, ns1, ns1 + 1)
        xx1[0] = 1
        xx2 = 1.0 / numpy.linspace(0, ns2, ns2 + 1)
        xx2[0] = 1
        fs1 = moments.Triallele.TriSpectrum(numpy.outer(xx1, xx1))
        fs2 = moments.Triallele.TriSpectrum(numpy.outer(xx2, xx2))
        J = moments.Triallele.Jackknife.calcJK_2(ns1)
        fs_jack = moments.Triallele.TriSpectrum(
            moments.Triallele.Numerics.reform(
                J.dot(moments.Triallele.Numerics.flatten(fs1)), ns2
            )
        )
        self.assertTrue(numpy.allclose(fs_jack, fs2, rtol=0.04))

    def test_neutral(self):
        ns = 30
        xx = 1.0 / numpy.linspace(0, ns, ns + 1)
        xx[0] = 1.0
        exact = moments.Triallele.TriSpectrum(numpy.outer(xx, xx))
        fs = moments.Triallele.TriSpectrum(numpy.zeros((ns + 1, ns + 1)))
        fs.integrate(1.0, 20.0)
        self.assertTrue(numpy.allclose(fs, exact))

    def test_selection(self):
        ns = 30
        cached = moments.Triallele.TriSpectrum.from_file(
            os.path.join(os.path.dirname(__file__), "test_files/triallele_selection.fs")
        )
        sA, hA = -1.0, 0.5
        sB, hB = -1.0, 0.5
        # from cached, integrate T=1 to make sure stays constant
        gammas = (2 * sA, 2 * sA * hA, 2 * sB, 2 * sB * hB, 2 * sA * hA + 2 * sB * hB)
        fs = copy.deepcopy(cached)
        fs.integrate(1.0, 1.0, gammas=gammas)
        self.assertTrue(
            numpy.allclose(fs, cached, rtol=5e-4)
        )


suite = unittest.TestLoader().loadTestsFromTestCase(TrialleleTestCase)

if __name__ == "__main__":
    unittest.main()
