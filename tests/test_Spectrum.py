import os
import unittest

import numpy as np
import scipy.special
import moments
import pickle
import time


class TestLoadDump(unittest.TestCase):
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
        data = np.random.rand(3, 3)

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
        data = np.random.rand(3, 3)

        fsin = moments.Spectrum(data)
        fsin.to_file(filename, comment_lines=commentsin)

        # Read the file.
        fsout, commentsout = moments.Spectrum.from_file(filename, return_comments=True)
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assertTrue(np.allclose(fsout.data, fsin.data))
        self.assertTrue(np.all(fsout.mask == fsin.mask))
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
        self.assertTrue(np.allclose(fsout.data, fsin.data))
        self.assertTrue(np.all(fsout.mask == fsin.mask))
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
        self.assertTrue(np.allclose(fsout.data, fsin.data))
        self.assertTrue(np.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)

    def test_pickle(self):
        """
        Saving spectrum to file.
        """
        comments = ["comment 1", "comment 2"]
        filename = "test.p"
        data = np.random.rand(3, 3)

        fs = moments.Spectrum(data)

        with open(filename, "wb") as f:
            pickle.dump(fs, f)
        os.remove(filename)

    def test_unpickle(self):
        """
        Loading spectrum from file.
        """
        commentsin = ["comment 1", "comment 2"]
        filename = "test.p"
        data = np.random.rand(3, 3)

        fsin = moments.Spectrum(data)

        with open(filename, "wb") as f:
            pickle.dump(fsin, f)

        # Read the file.
        with open(filename, "rb") as f:
            fsout = pickle.load(f)
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assertTrue(np.allclose(fsout.data, fsin.data))
        self.assertTrue(np.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)

        #
        # Now test a file with folding and masking
        #
        fsin = moments.Spectrum(data).fold()
        fsin.mask[0, 1] = True

        with open(filename, "wb") as f:
            pickle.dump(fsin, f)

        # Read the file.
        with open(filename, "rb") as f:
            fsout = pickle.load(f)
        os.remove(filename)
        # Ensure that fs was read correctly.
        self.assertTrue(np.allclose(fsout.data, fsin.data))
        self.assertTrue(np.all(fsout.mask == fsin.mask))
        self.assertEqual(fsout.folded, fsin.folded)

    def test_from_angsd(self):
        fname = os.path.join(
            os.path.dirname(__file__),
            "test_files/two_pop_angsd.82.82.sfs",
        )
        ns = [82, 82]
        fs = moments.Spectrum.from_angsd(fname, ns)
        self.assertFalse(fs.folded)
        self.assertTrue(fs.mask[0, 0])
        self.assertTrue(fs.mask[-1, -1])
        self.assertTrue(fs.pop_ids is None)

        pop_ids = ["A", "B"]
        fs = moments.Spectrum.from_angsd(fname, ns, pop_ids)
        self.assertEqual(fs.pop_ids[0], pop_ids[0])
        self.assertEqual(fs.pop_ids[1], pop_ids[1])

        fs = moments.Spectrum.from_angsd(fname, ns, folded=True)
        self.assertTrue(fs.folded)

        fs = moments.Spectrum.from_angsd(fname, ns, mask_corners=False)
        self.assertFalse(fs.mask[0, 0])
        self.assertFalse(fs.mask[-1, -1])

        fs = moments.Spectrum.from_angsd(fname, ns, mask_corners=False, folded=True)
        self.assertFalse(fs.mask[0, 0])
        self.assertTrue(fs.mask[-1, -1])

        with self.assertRaises(ValueError):
            fs = moments.Spectrum.from_angsd(fname, [41, 41])
        with self.assertRaises(ValueError):
            fs = moments.Spectrum.from_angsd(fname, ns, ["A"])
        with self.assertRaises(ValueError):
            fs = moments.Spectrum.from_angsd(fname, ns, ["A", "B", "C"])


class TestFolding(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_folding(self):
        """
        Folding a 2D spectrum.
        """
        data = np.reshape(np.arange(12), (3, 4))
        fs = moments.Spectrum(data)
        ff = fs.fold()

        # Ensure no SNPs have gotten lost.
        self.assertAlmostEqual(fs.sum(), ff.sum(), 6)
        self.assertAlmostEqual(fs.data.sum(), ff.data.sum(), 6)
        # Ensure that the empty entries are actually empty.
        self.assertTrue(np.all(ff.data[::-1] == np.tril(ff.data[::-1])))

        # This turns out to be the correct result.
        correct = np.tri(4)[::-1][-3:] * 11
        self.assertTrue(np.allclose(correct, ff.data))

    def test_ambiguous_folding(self):
        """
        Test folding when the minor allele is ambiguous.
        """
        data = np.zeros((4, 4))
        # Both these entries correspond to a an allele seen in 3 of 6 samples.
        # So the minor allele is ambiguous. In this case, we average the two
        # possible assignments.
        data[0, 3] = 1
        data[3, 0] = 3
        fs = moments.Spectrum(data)
        ff = fs.fold()

        correct = np.zeros((4, 4))
        correct[0, 3] = correct[3, 0] = 2
        self.assertTrue(np.allclose(correct, ff.data))

    def test_masked_folding(self):
        """
        Test folding with masked entries.
        """
        data = np.zeros((5, 6))
        fs = moments.Spectrum(data)
        # This folds to an entry that will already be masked.
        fs.mask[1, 2] = True
        # This folds to (1,1), which needs to be masked.
        fs.mask[3, 4] = True
        ff = fs.fold()
        # Ensure that all those are masked.
        for entry in [(1, 2), (3, 4), (1, 1)]:
            self.assertTrue(ff.mask[entry])

    def test_folded_slices(self):
        ns = (3, 4)
        fs1 = moments.Spectrum(np.random.rand(*ns))
        folded1 = fs1.fold()

        self.assertTrue(fs1[:].folded == False)
        self.assertTrue(folded1[:].folded == True)

        self.assertTrue(fs1[0].folded == False)
        self.assertTrue(folded1[1].folded == True)

        self.assertTrue(fs1[:, 0].folded == False)
        self.assertTrue(folded1[:, 1].folded == True)

    def test_folded_arithmetic(self):
        """
        Test that arithmetic operations respect and propogate .folded attribute.
        """
        # Disable logging of warnings because arithmetic may generate Spectra
        # with entries < 0, but we don't care at this point.
        import logging

        moments.Spectrum_mod.logger.setLevel(logging.ERROR)

        ns = (3, 4)
        fs1 = moments.Spectrum(np.random.uniform(size=ns))
        fs2 = moments.Spectrum(np.random.uniform(size=ns))

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

        arr = np.random.uniform(size=ns)
        marr = np.random.uniform(size=ns)

        # I found some difficulties with multiplication by np.float64, so I
        # want to explicitly test this case.
        npfloat = np.float64(2.0)

        for op in lst:
            # Check that binary operations propogate folding status.
            # Need to check cases both on right-hand-side of operator and
            # left-hand-side

            # Note that np.power(2.0,fs2) does not properly propagate type
            # or status. I'm not sure how to fix this.

            result = op(fs1, fs2)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs1.mask))

            result = op(fs1, 2.0)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs1.mask))

            result = op(2.0, fs2)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs2.mask))

            result = op(fs1, npfloat)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs1.mask))

            result = op(npfloat, fs2)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs2.mask))

            result = op(fs1, arr)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs1.mask))

            result = op(arr, fs2)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs2.mask))

            result = op(fs1, marr)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs1.mask))

            result = op(marr, fs2)
            self.assertFalse(result.folded)
            self.assertTrue(np.all(result.mask == fs2.mask))

            # Now with folded Spectra

            result = op(folded1, folded2)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded1.mask))

            result = op(folded1, 2.0)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded1.mask))

            result = op(2.0, folded2)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded2.mask))

            result = op(folded1, npfloat)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded1.mask))

            result = op(npfloat, folded2)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded2.mask))

            result = op(folded1, arr)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded1.mask))

            result = op(arr, folded2)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded2.mask))

            result = op(folded1, marr)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded1.mask))

            result = op(marr, folded2)
            self.assertTrue(result.folded)
            self.assertTrue(np.all(result.mask == folded2.mask))

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
                self.assertTrue(np.all(fs1.mask == fs1origmask))

                op(fs1, 2.0)
                self.assertFalse(fs1.folded)
                self.assertTrue(np.all(fs1.mask == fs1origmask))

                op(fs1, npfloat)
                self.assertFalse(fs1.folded)
                self.assertTrue(np.all(fs1.mask == fs1origmask))

                op(fs1, arr)
                self.assertFalse(fs1.folded)
                self.assertTrue(np.all(fs1.mask == fs1origmask))

                op(fs1, marr)
                self.assertFalse(fs1.folded)
                self.assertTrue(np.all(fs1.mask == fs1origmask))

                # Now folded Spectra
                folded1origmask = folded1.mask.copy()

                op(folded1, folded2)
                self.assertTrue(folded1.folded)
                self.assertTrue(np.all(folded1.mask == folded1origmask))

                op(folded1, 2.0)
                self.assertTrue(folded1.folded)
                self.assertTrue(np.all(folded1.mask == folded1origmask))

                op(folded1, npfloat)
                self.assertTrue(folded1.folded)
                self.assertTrue(np.all(folded1.mask == folded1origmask))

                op(folded1, arr)
                self.assertTrue(folded1.folded)
                self.assertTrue(np.all(folded1.mask == folded1origmask))

                op(folded1, marr)
                self.assertTrue(folded1.folded)
                self.assertTrue(np.all(folded1.mask == folded1origmask))

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
        fs = moments.Spectrum(np.random.uniform(size=ns))
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

    def test_folding_masked_corners(self):
        fs = moments.Spectrum(np.random.rand(5, 5), mask_corners=False)
        fs = fs.fold()
        self.assertTrue(fs.mask[-1, -1])
        self.assertFalse(fs.mask[0, 0])


class TestMarginalize(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_marginalize(self):
        ns = (7, 8, 6)

        fs = moments.Spectrum(np.random.uniform(size=ns))
        folded = fs.fold()

        marg1 = fs.marginalize([1])
        # Do manual marginalization.
        manual = moments.Spectrum(fs.data.sum(axis=1))

        # Check that these are equal in the unmasked entries.
        self.assertTrue(
            np.allclose(
                np.where(marg1.mask, 0, marg1.data),
                np.where(manual.mask, 0, manual.data),
            )
        )

        # Check folded Spectrum objects. I should get the same result if I
        # marginalize then fold, as if I fold then marginalize.
        mf1 = marg1.fold()
        mf2 = folded.marginalize([1])
        self.assertTrue(np.allclose(mf1, mf2))

    def test_fold_unmasked(self):
        fs = moments.Demographics2D.snm([10, 10])
        fs.mask.flat[0] = False
        fs_marg = fs.marginalize([0])
        self.assertEqual(sum(fs_marg.mask), 2)

        fs.mask.flat[0] = True
        fs.mask.flat[-1] = False
        fs_marg = fs.marginalize([0])
        self.assertEqual(sum(fs_marg.mask), 2)

        fs.mask.flat[0] = False
        fs_marg = fs.marginalize([0])
        self.assertEqual(sum(fs_marg.mask), 0)


class TestProjection(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_projection(self):
        # Test that projecting a multi-dimensional Spectrum succeeds
        ns = (7, 8, 6)
        fs = moments.Spectrum(np.random.uniform(size=ns))
        p = fs.project([3, 4, 5])
        # Also that we don't lose any data
        self.assertAlmostEqual(fs.data.sum(), p.data.sum())

        # Check that when I project an equilibrium spectrum, I get back an
        # equilibrium spectrum
        fs = moments.Spectrum(1.0 / np.arange(100))
        p = fs.project([17])
        self.assertTrue(np.allclose(p[1:-1], 1.0 / np.arange(1, len(p) - 1)))

        # Check that masked values are propagated correctly.
        fs = moments.Spectrum(1.0 / np.arange(20))
        # All values with 3 or fewer observed should be masked.
        fs.mask[3] = True
        p = fs.project([10])
        self.assertTrue(np.all(p.mask[:4]))

        # Check that masked values are propagated correctly.
        fs = moments.Spectrum(1.0 / np.arange(20))
        fs.mask[-3] = True
        # All values with 3 or fewer observed should be masked.
        p = fs.project([10])
        self.assertTrue(np.all(p.mask[-3:]))

        # A more complicated two dimensional projection problem...
        fs = moments.Spectrum(np.random.uniform(size=(9, 7)))
        fs.mask[2, 3] = True
        p = fs.project([4, 4])
        self.assertTrue(np.all(p.mask[:3, 1:4]))

        # Test that projecting a folded multi-dimensional Spectrum succeeds
        # Should get the same result if I fold then project as if I project
        # then fold.
        ns = (7, 8, 6)
        fs = moments.Spectrum(np.random.uniform(size=ns))
        fs.mask[2, 3, 1] = True
        folded = fs.fold()

        p = fs.project([3, 4, 5])
        pf1 = p.fold()
        pf2 = folded.project([3, 4, 5])

        # Check equality
        self.assertTrue(np.all(pf1.mask == pf2.mask))
        self.assertTrue(np.allclose(pf1.data, pf2.data))


class TestAdmixture(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

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

        fs = moments.Spectrum(np.random.uniform(size=ns))

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
        self.assertTrue(np.allclose(fs_1_into_2, fs_sequential.transpose((0, 2, 1))))


class TestSwapAxes(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_swap_ids(self):
        ns = (8, 5)
        fs = moments.Spectrum(np.random.uniform(size=ns))
        fs.pop_ids = ["A", "B"]
        fs_swap = fs.swap_axes(0, 1)
        self.assertTrue(np.all(fs_swap.data == fs.data.T))
        self.assertTrue(fs_swap.pop_ids[0] == "B")
        self.assertTrue(fs_swap.pop_ids[1] == "A")


class TestSplitFunction(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_split(self):
        fs = moments.Spectrum(np.ones(11))
        with self.assertRaises(ValueError):
            fs.split(0, 5, 6)
        with self.assertRaises(ValueError):
            fs.split(1, 5, 5)
        with self.assertRaises(ValueError):
            fs.split(0, 5, 5, new_ids=["A", "B"])
        fs = fs.fold()
        with self.assertRaises(ValueError):
            fs.split(0, 5, 5)
        fs = moments.Spectrum(np.ones((5, 5, 5)))
        with self.assertRaises(ValueError):
            fs.split(-1, 2, 2)
        with self.assertRaises(ValueError):
            fs.split(0, 3, 3)
        with self.assertRaises(ValueError):
            fs.split(3, 2, 2)

    def test_split_1D(self):
        fs = moments.Spectrum(np.ones(11))
        out = fs.split(0, 5, 5)
        self.assertTrue(np.all(np.array(out.shape) == np.array([6, 6])))
        out = fs.split(0, 2, 8)
        self.assertTrue(np.all(np.array(out.shape) == np.array([3, 9])))
        fs.pop_ids = ["anc"]
        out = fs.split(0, 5, 5)
        self.assertTrue(out.pop_ids is None)
        out = fs.split(0, 5, 5, new_ids=["A", "B"])
        self.assertEqual(out.pop_ids[0], "A")
        self.assertEqual(out.pop_ids[1], "B")

    def test_split_2D(self):
        fs = moments.Spectrum(np.ones((11, 11)))
        out = fs.split(0, 5, 5)
        self.assertTrue(np.all(np.array(out.shape) == [6, 11, 6]))
        out = fs.split(1, 1, 9)
        self.assertTrue(np.all(np.array(out.shape) == [11, 2, 10]))

    def test_split_pop_ids(self):
        fs = moments.Spectrum(np.ones((11, 11, 11)))
        fs.pop_ids = ["A", "B", "C"]
        out = fs.split(1, 5, 5, new_ids=["X", "Y"])
        self.assertTrue(
            np.all([i == j for i, j in zip(out.pop_ids, ["A", "X", "C", "Y"])])
        )


class TestAdmixFunction(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_admix(self):
        fs = moments.Spectrum(np.ones((11, 11)))
        with self.assertRaises(ValueError):
            fs.admix(0, 1, 20, 0.5)
        with self.assertRaises(ValueError):
            fs.admix(0, 1, 10, 1.5)
        with self.assertRaises(ValueError):
            fs.admix(0, 1, 5, 0.5, new_id="X")
        with self.assertRaises(ValueError):
            fs.admix(0, 0, 5, 0.5)
        with self.assertRaises(ValueError):
            fs.admix(1, 2, 5, 0.5)

    def test_admix_2D(self):
        fs = moments.Spectrum(np.ones((11, 11)))
        out = fs.admix(0, 1, 10, 0.25)
        self.assertEqual(out.Npop, 1)
        self.assertEqual(out.sample_sizes[0], 10)
        out = fs.admix(0, 1, 4, 0.5)
        self.assertEqual(out.Npop, 3)
        self.assertTrue(np.all([i == j for i, j in zip(out.shape, (7, 7, 5))]))
        fs.pop_ids = ["A", "B"]
        out = fs.admix(0, 1, 5, 0.25, new_id="C")
        self.assertTrue(out.pop_ids[2] == "C")
        fs = moments.Spectrum(np.ones((6, 11)))
        fs.pop_ids = ["A", "B"]
        out = fs.admix(0, 1, 5, 0.5, new_id="C")
        self.assertEqual(out.Npop, 2)
        self.assertTrue(np.all([i == j for i, j in zip(out.pop_ids, ["B", "C"])]))


class TestPulseMigrateFunction(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_pulse(self):
        fs = moments.Spectrum(np.ones((11, 11, 11)))
        with self.assertRaises(ValueError):
            fs.pulse_migrate(0, 1, 15, 0.1)
        with self.assertRaises(ValueError):
            fs.pulse_migrate(0, 3, 5, 0.01)
        with self.assertRaises(ValueError):
            fs.pulse_migrate(0, 0, 5, 0.01)

    def test_pulse_migrate_func(self):
        fs = moments.Spectrum(np.ones((11, 11, 11)))
        fs.pop_ids = ["A", "B", "C"]
        out = fs.pulse_migrate(0, 1, 5, 0.01)
        self.assertTrue(np.all([i == j for i, j in zip(out.sample_sizes, [5, 10, 10])]))
        self.assertTrue(np.all([i == j for i, j in zip(out.pop_ids, fs.pop_ids)]))


class TestBranchFunction(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_bad_branch(self):
        fs = moments.Spectrum(np.ones((11, 11)))
        with self.assertRaises(ValueError):
            fs.branch(2, 5)
        with self.assertRaises(ValueError):
            fs.branch(-1, 5)
        with self.assertRaises(ValueError):
            fs.branch(1, 12)
        with self.assertRaises(ValueError):
            fs.branch(0, 5, new_id="a")

    def test_branch_func(self):
        fs = moments.Spectrum(np.ones((11, 11)))
        fs.pop_ids = ["A", "B"]
        out0 = fs.branch(0, 4, new_id="C")
        out1 = fs.branch(1, 6, new_id="C")
        self.assertTrue(np.all([i == j for i, j in zip(out0.sample_sizes, [6, 10, 4])]))
        self.assertTrue(np.all([i == j for i, j in zip(out0.pop_ids, ["A", "B", "C"])]))
        self.assertTrue(np.all([i == j for i, j in zip(out1.sample_sizes, [10, 4, 6])]))
        self.assertTrue(np.all([i == j for i, j in zip(out1.pop_ids, ["A", "B", "C"])]))


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_poisson_sample(self):
        fs = moments.Demographics1D.snm([10])
        samp = fs.sample()
        self.assertEqual(samp.dtype, int)
        fs *= 10000
        samp2 = fs.sample()
        self.assertTrue(samp.S() < samp2.S())

    def test_fixed_size_sample(self):
        fs = moments.Demographics1D.snm([10])
        n = 10
        samp = fs.fixed_size_sample(n)
        self.assertEqual(samp.dtype, int)
        self.assertEqual(samp.S(), n)
        fs = moments.Demographics2D.snm([20, 20])
        n = 47
        samp = fs.fixed_size_sample(n)
        self.assertEqual(samp.dtype, int)
        self.assertEqual(samp.S(), n)

    def test_genotype_matrix(self):
        fs = moments.Spectrum(np.zeros(11))
        fs[2] = 1
        n_sites = 20
        G = fs.genotype_matrix(num_sites=n_sites)
        self.assertEqual(len(G), n_sites)
        self.assertEqual(len(G[0]), fs.sample_sizes[0])
        self.assertTrue(np.all(G.sum(axis=1) == 2))
        G = fs.genotype_matrix(num_sites=n_sites, diploid_genotypes=True)
        self.assertEqual(len(G), n_sites)
        self.assertEqual(len(G[0]), fs.sample_sizes[0] // 2)
        self.assertTrue(np.all(G.sum(axis=1) == 2))

        fs = moments.Demographics2D.snm([10, 20]) * 10
        G = fs.genotype_matrix()
        self.assertEqual(len(G[0]), fs.sample_sizes.sum())

    def test_bad_genotype_matrix_arguments(self):
        fs = moments.Demographics2D.snm([10, 10])
        with self.assertRaises(ValueError):
            G = fs.genotype_matrix(sample_sizes=[20, 10])
        with self.assertRaises(ValueError):
            G = fs.genotype_matrix(sample_sizes=[5, 20])
        with self.assertRaises(ValueError):
            G = fs.genotype_matrix(sample_sizes=[5])
        with self.assertRaises(ValueError):
            G = fs.genotype_matrix(sample_sizes=[5, 5, 5])

        fs = moments.Demographics1D.snm([11])
        with self.assertRaises(ValueError):
            G = fs.genotype_matrix(diploid_genotypes=True)
