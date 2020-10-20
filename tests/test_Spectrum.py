import os
import unittest

import numpy
import scipy.special
import moments
import pickle
import time


class SpectrumTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_to_file(self):
        """
        Saving spectrum to file.
        """
        comments = ["comment 1", "comment 2"]
        filename = "test.fs"
        data = numpy.random.rand(3, 3)

        fs = moments.Spectrum(data)

        fs.to_file(filename, comment_lines=comments)
        os.remove(filename)

        fs.to_file(filename, comment_lines=comments, foldmaskinfo=False)
        os.remove(filename)

    def test_from_file(self):
        """
        Loading spectrum from file.
        """
        commentsin = ["comment 1", "comment 2"]
        filename = "test.fs"
        data = numpy.random.rand(3, 3)

        fsin = moments.Spectrum(data)
        fsin.to_file(filename, comment_lines=commentsin)

        # Read the file.
        fsout, commentsout = moments.Spectrum.from_file(filename, return_comments=True)
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assert_(numpy.allclose(fsout.data, fsin.data))
        self.assert_(numpy.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)
        # Ensure comments were read correctly.
        for ii, line in enumerate(commentsin):
            self.assertEqual(line, commentsout[ii])

        # Test using old file format
        fsin.to_file(filename, comment_lines=commentsin, foldmaskinfo=False)

        # Read the file.
        fsout, commentsout = moments.Spectrum.from_file(filename, return_comments=True)
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assert_(numpy.allclose(fsout.data, fsin.data))
        self.assert_(numpy.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)
        # Ensure comments were read correctly.
        for ii, line in enumerate(commentsin):
            self.assertEqual(line, commentsout[ii])

        #
        # Now test a file with folding and masking
        #
        fsin = moments.Spectrum(data).fold()
        fsin.mask[0, 1] = True
        fsin.to_file(filename)

        fsout = moments.Spectrum.from_file(filename)
        os.remove(filename)

        # Ensure that fs was read correctly.
        self.assert_(numpy.allclose(fsout.data, fsin.data))
        self.assert_(numpy.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)

    def test_pickle(self):
        """
        Saving spectrum to file.
        """
        comments = ["comment 1", "comment 2"]
        filename = "test.p"
        data = numpy.random.rand(3, 3)

        fs = moments.Spectrum(data)

        pickle.dump(fs, open(filename, "wb"))
        os.remove(filename)

    def test_unpickle(self):
        """
        Loading spectrum from file.
        """
        commentsin = ["comment 1", "comment 2"]
        filename = "test.p"
        data = numpy.random.rand(3, 3)

        fsin = moments.Spectrum(data)

        pickle.dump(fsin, open(filename, "wb"))

        # Read the file.
        fsout = pickle.load(open(filename, "rb"))
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assert_(numpy.allclose(fsout.data, fsin.data))
        self.assert_(numpy.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)

        #
        # Now test a file with folding and masking
        #
        fsin = moments.Spectrum(data).fold()
        fsin.mask[0, 1] = True

        pickle.dump(fsin, open(filename, "wb"))

        # Read the file.
        fsout = pickle.load(open(filename, "rb"))
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assert_(numpy.allclose(fsout.data, fsin.data))
        self.assert_(numpy.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)

    def test_folding(self):
        """
        Folding a 2D spectrum.
        """
        data = numpy.reshape(numpy.arange(12), (3, 4))
        fs = moments.Spectrum(data)
        ff = fs.fold()

        # Ensure no SNPs have gotten lost.
        self.assertAlmostEqual(fs.sum(), ff.sum(), 6)
        self.assertAlmostEqual(fs.data.sum(), ff.data.sum(), 6)
        # Ensure that the empty entries are actually empty.
        self.assert_(numpy.all(ff.data[::-1] == numpy.tril(ff.data[::-1])))

        # This turns out to be the correct result.
        correct = numpy.tri(4)[::-1][-3:] * 11
        self.assert_(numpy.allclose(correct, ff.data))

    def test_ambiguous_folding(self):
        """
        Test folding when the minor allele is ambiguous.
        """
        data = numpy.zeros((4, 4))
        # Both these entries correspond to a an allele seen in 3 of 6 samples.
        # So the minor allele is ambiguous. In this case, we average the two
        # possible assignments.
        data[0, 3] = 1
        data[3, 0] = 3
        fs = moments.Spectrum(data)
        ff = fs.fold()

        correct = numpy.zeros((4, 4))
        correct[0, 3] = correct[3, 0] = 2
        self.assert_(numpy.allclose(correct, ff.data))

    def test_masked_folding(self):
        """
        Test folding when the minor allele is ambiguous.
        """
        data = numpy.zeros((5, 6))
        fs = moments.Spectrum(data)
        # This folds to an entry that will already be masked.
        fs.mask[1, 2] = True
        # This folds to (1,1), which needs to be masked.
        fs.mask[3, 4] = True
        ff = fs.fold()
        # Ensure that all those are masked.
        for entry in [(1, 2), (3, 4), (1, 1)]:
            self.assert_(ff.mask[entry])

    def test_folded_slices(self):
        ns = (3, 4)
        fs1 = moments.Spectrum(numpy.random.rand(*ns))
        folded1 = fs1.fold()

        self.assert_(fs1[:].folded == False)
        self.assert_(folded1[:].folded == True)

        self.assert_(fs1[0].folded == False)
        self.assert_(folded1[1].folded == True)

        self.assert_(fs1[:, 0].folded == False)
        self.assert_(folded1[:, 1].folded == True)

    def test_folded_arithmetic(self):
        """
        Test that arithmetic operations respect and propogate .folded attribute.
        """
        # Disable logging of warnings because arithmetic may generate Spectra
        # with entries < 0, but we don't care at this point.
        import logging

        moments.Spectrum_mod.logger.setLevel(logging.ERROR)

        ns = (3, 4)
        fs1 = moments.Spectrum(numpy.random.uniform(size=ns))
        fs2 = moments.Spectrum(numpy.random.uniform(size=ns))

        folded1 = fs1.fold()
        folded2 = fs2.fold()

        # We'll iterate through each of these arithmetic functions.
        try:
            from operator import (
                add,
                sub,
                mul,
                div,
                truediv,
                floordiv,
                pow,
                abs,
                pos,
                neg,
            )

            lst = [add, sub, mul, div, truediv, floordiv, pow]
        except:
            from operator import add, sub, mul, truediv, floordiv, pow, abs, pos, neg

            lst = [add, sub, mul, truediv, floordiv, pow]

        arr = numpy.random.uniform(size=ns)
        marr = numpy.random.uniform(size=ns)

        # I found some difficulties with multiplication by numpy.float64, so I
        # want to explicitly test this case.
        numpyfloat = numpy.float64(2.0)

        for op in lst:
            # Check that binary operations propogate folding status.
            # Need to check cases both on right-hand-side of operator and
            # left-hand-side

            # Note that numpy.power(2.0,fs2) does not properly propagate type
            # or status. I'm not sure how to fix this.

            result = op(fs1, fs2)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs1.mask))

            result = op(fs1, 2.0)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs1.mask))

            result = op(2.0, fs2)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs2.mask))

            result = op(fs1, numpyfloat)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs1.mask))

            result = op(numpyfloat, fs2)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs2.mask))

            result = op(fs1, arr)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs1.mask))

            result = op(arr, fs2)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs2.mask))

            result = op(fs1, marr)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs1.mask))

            result = op(marr, fs2)
            self.assertFalse(result.folded)
            self.assert_(numpy.all(result.mask == fs2.mask))

            # Now with folded Spectra

            result = op(folded1, folded2)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded1.mask))

            result = op(folded1, 2.0)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded1.mask))

            result = op(2.0, folded2)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded2.mask))

            result = op(folded1, numpyfloat)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded1.mask))

            result = op(numpyfloat, folded2)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded2.mask))

            result = op(folded1, arr)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded1.mask))

            result = op(arr, folded2)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded2.mask))

            result = op(folded1, marr)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded1.mask))

            result = op(marr, folded2)
            self.assertTrue(result.folded)
            self.assert_(numpy.all(result.mask == folded2.mask))

            # Check that exceptions are properly raised when folding status
            # differs
            self.assertRaises(ValueError, op, fs1, folded2)
            self.assertRaises(ValueError, op, folded1, fs2)

        for op in [abs, pos, neg, scipy.special.gammaln]:
            # Check that unary operations propogate folding status.
            result = op(fs1)
            self.assertFalse(result.folded)
            result = op(folded1)
            self.assertTrue(result.folded)

        try:
            # The in-place methods aren't in operator in python 2.4...
            from operator import iadd, isub, imul, idiv, itruediv, ifloordiv, ipow

            for op in [iadd, isub, imul, idiv, itruediv, ifloordiv, ipow]:
                fs1origmask = fs1.mask.copy()

                # Check that in-place operations preserve folding status.
                op(fs1, fs2)
                self.assertFalse(fs1.folded)
                self.assert_(numpy.all(fs1.mask == fs1origmask))

                op(fs1, 2.0)
                self.assertFalse(fs1.folded)
                self.assert_(numpy.all(fs1.mask == fs1origmask))

                op(fs1, numpyfloat)
                self.assertFalse(fs1.folded)
                self.assert_(numpy.all(fs1.mask == fs1origmask))

                op(fs1, arr)
                self.assertFalse(fs1.folded)
                self.assert_(numpy.all(fs1.mask == fs1origmask))

                op(fs1, marr)
                self.assertFalse(fs1.folded)
                self.assert_(numpy.all(fs1.mask == fs1origmask))

                # Now folded Spectra
                folded1origmask = folded1.mask.copy()

                op(folded1, folded2)
                self.assertTrue(folded1.folded)
                self.assert_(numpy.all(folded1.mask == folded1origmask))

                op(folded1, 2.0)
                self.assertTrue(folded1.folded)
                self.assert_(numpy.all(folded1.mask == folded1origmask))

                op(folded1, numpyfloat)
                self.assertTrue(folded1.folded)
                self.assert_(numpy.all(folded1.mask == folded1origmask))

                op(folded1, arr)
                self.assertTrue(folded1.folded)
                self.assert_(numpy.all(folded1.mask == folded1origmask))

                op(folded1, marr)
                self.assertTrue(folded1.folded)
                self.assert_(numpy.all(folded1.mask == folded1origmask))

                # Check that exceptions are properly raised.
                self.assertRaises(ValueError, op, fs1, folded2)
                self.assertRaises(ValueError, op, folded1, fs2)
        except ImportError:
            pass

        # Restore logging of warnings
        moments.Spectrum_mod.logger.setLevel(logging.WARNING)

    def test_unfolding(self):
        ns = (3, 4)

        # We add some unusual masking.
        fs = moments.Spectrum(numpy.random.uniform(size=ns))
        fs.mask[0, 1] = fs.mask[1, 1] = True

        folded = fs.fold()
        unfolded = folded.unfold()

        # Check that it was properly recorded
        self.assertFalse(unfolded.folded)

        # Check that no data was lost
        self.assertAlmostEqual(fs.data.sum(), folded.data.sum())
        self.assertAlmostEqual(fs.data.sum(), unfolded.data.sum())

        # Note that fs.sum() need not be equal to folded.sum(), if fs had
        # some masked values.
        self.assertAlmostEqual(folded.sum(), unfolded.sum())

        # Check that the proper entries are masked.
        self.assertTrue(unfolded.mask[0, 1])
        self.assertTrue(unfolded.mask[(ns[0] - 1), (ns[1] - 1) - 1])
        self.assertTrue(unfolded.mask[1, 1])
        self.assertTrue(unfolded.mask[(ns[0] - 1) - 1, (ns[1] - 1) - 1])

    def test_marginalize(self):
        ns = (7, 8, 6)

        fs = moments.Spectrum(numpy.random.uniform(size=ns))
        folded = fs.fold()

        marg1 = fs.marginalize([1])
        # Do manual marginalization.
        manual = moments.Spectrum(fs.data.sum(axis=1))

        # Check that these are equal in the unmasked entries.
        self.assert_(
            numpy.allclose(
                numpy.where(marg1.mask, 0, marg1.data),
                numpy.where(manual.mask, 0, manual.data),
            )
        )

        # Check folded Spectrum objects. I should get the same result if I
        # marginalize then fold, as if I fold then marginalize.
        mf1 = marg1.fold()
        mf2 = folded.marginalize([1])
        self.assert_(numpy.allclose(mf1, mf2))

    def test_projection(self):
        # Test that projecting a multi-dimensional Spectrum succeeds
        ns = (7, 8, 6)
        fs = moments.Spectrum(numpy.random.uniform(size=ns))
        p = fs.project([3, 4, 5])
        # Also that we don't lose any data
        self.assertAlmostEqual(fs.data.sum(), p.data.sum())

        # Check that when I project an equilibrium spectrum, I get back an
        # equilibrium spectrum
        fs = moments.Spectrum(1.0 / numpy.arange(100))
        p = fs.project([17])
        self.assert_(numpy.allclose(p[1:-1], 1.0 / numpy.arange(1, len(p) - 1)))

        # Check that masked values are propagated correctly.
        fs = moments.Spectrum(1.0 / numpy.arange(20))
        # All values with 3 or fewer observed should be masked.
        fs.mask[3] = True
        p = fs.project([10])
        self.assert_(numpy.all(p.mask[:4]))

        # Check that masked values are propagated correctly.
        fs = moments.Spectrum(1.0 / numpy.arange(20))
        fs.mask[-3] = True
        # All values with 3 or fewer observed should be masked.
        p = fs.project([10])
        self.assert_(numpy.all(p.mask[-3:]))

        # A more complicated two dimensional projection problem...
        fs = moments.Spectrum(numpy.random.uniform(size=(9, 7)))
        fs.mask[2, 3] = True
        p = fs.project([4, 4])
        self.assert_(numpy.all(p.mask[:3, 1:4]))

        # Test that projecting a folded multi-dimensional Spectrum succeeds
        # Should get the same result if I fold then project as if I project
        # then fold.
        ns = (7, 8, 6)
        fs = moments.Spectrum(numpy.random.uniform(size=ns))
        fs.mask[2, 3, 1] = True
        folded = fs.fold()

        p = fs.project([3, 4, 5])
        pf1 = p.fold()
        pf2 = folded.project([3, 4, 5])

        # Check equality
        self.assert_(numpy.all(pf1.mask == pf2.mask))
        self.assert_(numpy.allclose(pf1.data, pf2.data))

    def test_admix(self):
        # Test that projecting a multi-dimensional Spectrum succeeds
        ns = (25, 8, 6)
        m_12 = 0.5  # 1 towards 2
        nu = 17

        target_n1 = 7
        target_n2 = 7
        target_n3 = 5

        n1_sequential = target_n1 + nu
        n1_exact = target_n1 + target_n2

        project_dp = [target_n1 + target_n2, target_n2, target_n3]
        project_seq = [n1_sequential, target_n2, target_n3]

        fs = moments.Spectrum(numpy.random.uniform(size=ns))

        # admix
        fs_1_into_2 = moments.Manips.admix_into_new(
            fs.project(project_dp),
            dimension1=0,
            dimension2=1,
            n_lineages=target_n2,
            m1=m_12,
        )

        #
        fs = fs.project(project_seq)
        fs_sequential = moments.Manips.admix_inplace(
            fs,
            source_population_index=0,
            target_population_index=1,
            keep_1=target_n1,
            m1=m_12,
        )

        # Also that we don't lose any data
        self.assertTrue(numpy.allclose(fs_1_into_2, fs_sequential.transpose((0, 2, 1))))


suite = unittest.TestLoader().loadTestsFromTestCase(SpectrumTestCase)
