"""
Contains two locus spectrum object
"""
import logging

logging.basicConfig()
logger = logging.getLogger("TLSpectrum_mod")

import os
import copy
import numpy as np
from . import Numerics, Integration, Util
import scipy.special


class TLSpectrum(np.ma.masked_array):
    """
    Represents a two locus frequency spectrum.

    :param array data: The frequency spectrum data, which has shape
        (n+1)-by-(n+1)-by-(n+1) where n is the sample size.
    :param array mask: An optional array of the same size as data. 'True' entries
        in this array are masked in the TLSpectrum.
    :param bool mask_infeasible: If True, mask all bins for frequencies that cannot
        occur, e.g. i + j > n. Defaults to True.
    :param bool mask_fixed: If True, mask the fixed bins. Defaults to True.
    :param bool data_folded: If True, it is assumed that the input data is folded
        for the major and minor derived alleles
    :param bool check_folding: If True and data_folded=True, the data and
        mask will be checked to ensure they are consistent.
    """

    def __new__(
        subtype,
        data,
        mask=np.ma.nomask,
        mask_infeasible=True,
        mask_fixed=False,
        data_folded=None,
        check_folding=True,
        dtype=float,
        copy=True,
        fill_value=np.nan,
        keep_mask=True,
        shrink=True,
    ):
        data = np.asanyarray(data)

        if mask is np.ma.nomask:
            mask = np.ma.make_mask_none(data.shape)

        subarr = np.ma.masked_array(
            data,
            mask=mask,
            dtype=dtype,
            copy=copy,
            fill_value=fill_value,
            keep_mask=True,
            shrink=True,
        )
        subarr = subarr.view(subtype)
        if hasattr(data, "folded"):
            if data_folded is None or data_folded == data.folded:
                subarr.folded = data.folded
            elif data_folded != data.folded:
                raise ValueError(
                    "Data does not have same folding status as "
                    "was called for in Spectrum constructor."
                )
        elif data_folded is not None:
            subarr.folded = data_folded
        else:
            subarr.folded = False

        if mask_infeasible:
            subarr.mask_infeasible()

        if mask_fixed:
            subarr.mask_fixed()

        return subarr

    def __array_finalize__(self, obj):
        if obj is None:
            return
        np.ma.masked_array.__array_finalize__(self, obj)
        self.folded = getattr(obj, "folded", "unspecified")

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, context, return_scalar)
        result.folded = self.folded
        return result

    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, "folded"):
            self.folded = obj.folded

    # masked_array has priority 15.
    __array_priority__ = 20

    def __repr__(self):
        return "TLSpectrum(%s, folded=%s)" % (str(self), str(self.folded))

    def mask_infeasible(self):
        """
        Mask any infeasible entries.
        """
        ns = len(self) - 1
        # mask entries with i+j+k > ns
        for ii in range(len(self)):
            for jj in range(len(self)):
                for kk in range(len(self)):
                    if ii + jj + kk > ns:
                        self.mask[ii, jj, kk] = True

    def mask_fixed(self):
        """
        Mask all infeasible entries, as well as any where both sites are not
        segregating.
        """
        ns = len(self) - 1
        # mask fixed entries
        self.mask[0, 0, 0] = True
        self.mask[0, 0, -1] = True
        self.mask[0, -1, 0] = True
        self.mask[-1, 0, 0] = True
        # mask entries with i+j+k > ns
        for ii in range(len(self)):
            for jj in range(len(self)):
                for kk in range(len(self)):
                    if ii + jj + kk > ns:
                        self.mask[ii, jj, kk] = True

        # mask fA = 0 and fB = 0
        for ii in range(len(self)):
            self.mask[ii, ns - ii, 0] = True
            self.mask[ii, 0, ns - ii] = True

        self.mask[0, :, 0] = True
        self.mask[0, 0, :] = True

    def unfold(self):
        """
        Remove folding from the spectrum.
        """
        if not self.folded:
            raise ValueError("Input Spectrum is not folded.")
        data = self.data
        unfolded = TLSpectrum(data, mask_infeasible=True)
        return unfolded

    def _get_sample_size(self):
        return np.asarray(self.shape)[0] - 1

    sample_size = property(_get_sample_size)

    def left(self):
        """
        The marginal allele frequency spectrum at the left locus.
        """
        n = len(self) - 1
        fl = np.zeros(n + 1)
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    fl[ii + jj] += self[ii, jj, kk]
        return fl

    def right(self):
        """
        The marginal AFS at the right locus.
        """
        n = len(self) - 1
        fr = np.zeros(n + 1)
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    fr[ii + kk] += self[ii, jj, kk]
        return fr

    def ancestral_misid(self, p):
        """
        Return a new SFS with a given ancestral misidentification, p.

        :param p: The rate of ancestral state misidentification.
        """
        if p == 0:
            return (1 - p) * self
        elif p > 1 or p < 0:
            raise ValueError("probability of misid must be between 0 and 1")

        F_new = (1 - p) ** 2 * self
        for ii in range(self.sample_size + 1):
            for jj in range(self.sample_size + 1 - ii):
                for kk in range(self.sample_size + 1 - ii - jj):
                    ll = self.sample_size - ii - jj - kk
                    # left misid, right correct
                    F_new.data[kk, ll, ii] += p * (1 - p) * self.data[ii, jj, kk]
                    # right misid, left correct
                    F_new.data[jj, ii, ll] += (1 - p) * p * self.data[ii, jj, kk]
                    # both misid
                    F_new.data[ll, kk, jj] += p ** 2 * self.data[ii, jj, kk]
        return F_new

    # statistics, called from Util

    def S(self, nA=None, nB=None):
        """
        Return the sum of probabilities over all variable two-locus entries
        in the spectrum.
        """
        if nA is not None and (nA <= 0 or nA >= self.sample_size):
            raise ValueError(f"nA must be between 1 and {self.sample_size - 1}")
        if nB is not None and (nB <= 0 or nB >= self.sample_size):
            raise ValueError(f"nB must be between 1 and {self.sample_size - 1}")
        return Util.compute_S(self, nA, nB)

    def D(self, proj=True, nA=None, nB=None):
        """
        Return the expectation of :math:`D` from the spectrum.

        :param proj: If True, use the unbiased estimator from downsampling. If False,
            use naive maximum likelihood estimates for frequency.
        :param nA: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the left locus.
        :param nB: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the right locus.
        """
        if nA is not None and (nA <= 0 or nA >= self.sample_size):
            raise ValueError(f"nA must be between 1 and {self.sample_size - 1}")
        if nB is not None and (nB <= 0 or nB >= self.sample_size):
            raise ValueError(f"nB must be between 1 and {self.sample_size - 1}")
        return Util._compute_D(self, proj, nA, nB)

    def D2(self, proj=True, nA=None, nB=None):
        """
        Return the expectation of :math:`D^2` from the spectrum.

        :param proj: If True, use the unbiased estimator from downsampling. If False,
            use naive maximum likelihood estimates for frequency.
        :param nA: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the left locus.
        :param nB: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the right locus.
        """
        if nA is not None and (nA <= 0 or nA >= self.sample_size):
            raise ValueError(f"nA must be between 1 and {self.sample_size - 1}")
        if nB is not None and (nB <= 0 or nB >= self.sample_size):
            raise ValueError(f"nB must be between 1 and {self.sample_size - 1}")
        return Util._compute_D2(self, proj, nA, nB)

    def Dz(self, proj=True, nA=None, nB=None):
        """
        Compute the expectation of :math:`Dz = D(1-2p)(1-2q)` from the spectrum.

        :param proj: If True, use the unbiased estimator from downsampling. If False,
            use naive maximum likelihood estimates for frequency.
        :param nA: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the left locus.
        :param nB: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the right locus.
        """
        if nA is not None and (nA <= 0 or nA >= self.sample_size):
            raise ValueError(f"nA must be between 1 and {self.sample_size - 1}")
        if nB is not None and (nB <= 0 or nB >= self.sample_size):
            raise ValueError(f"nB must be between 1 and {self.sample_size - 1}")
        return Util._compute_Dz(self, proj, nA, nB)

    def pi2(self, proj=True, nA=None, nB=None):
        """
        Return the expectation of :math:`\\pi_2 = p(1-p)q(1-q)` from the spectrum.

        :param proj: If True, use the unbiased estimator from downsampling. If False,
            use naive maximum likelihood estimates for frequency.
        :param nA: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the left locus.
        :param nB: If None, the average is computed over all frequencies. If given,
            condition on the given allele count for the right locus.
        """
        if nA is not None and (nA <= 0 or nA >= self.sample_size):
            raise ValueError(f"nA must be between 1 and {self.sample_size - 1}")
        if nB is not None and (nB <= 0 or nB >= self.sample_size):
            raise ValueError(f"nB must be between 1 and {self.sample_size - 1}")
        return Util._compute_pi2(self, proj, nA, nB)

    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, mask_infeasible=True, return_comments=False):
        """
        Read frequency spectrum from file.

        :param str fid: String with file name to read from or an open file object.
        :param bool mask_infeasible: If True, mask the infeasible entries in the
            two locus spectrum.
        :param bool return_comments: If true, the return value is (fs, comments), where
            comments is a list of strings containing the comments from the file.
        """
        newfile = False
        # Try to read from fid. If we can't, assume it's something that we can
        # use to open a file.
        if not hasattr(fid, "read"):
            newfile = True
            fid = open(fid, "r")

        line = fid.readline()
        # Strip out the comments
        comments = []
        while line.startswith("#"):
            comments.append(line[1:].strip())
            line = fid.readline()

        # Read the shape of the data
        shape, folded = line.split()
        shape = [int(shape) + 1, int(shape) + 1, int(shape) + 1]

        data = np.fromstring(fid.readline().strip(), count=np.prod(shape), sep=" ")
        # fromfile returns a 1-d array. Reshape it to the proper form.
        data = data.reshape(*shape)

        maskline = fid.readline().strip()
        mask = np.fromstring(maskline, count=np.prod(shape), sep=" ")
        mask = mask.reshape(*shape)

        if folded == "folded":
            folded = True
        else:
            folded = False

        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        fs = TLSpectrum(data, mask, mask_infeasible, data_folded=folded)
        if not return_comments:
            return fs
        else:
            return fs, comments

    def to_file(self, fid, precision=16, comment_lines=[], foldmaskinfo=True):
        """
        Write frequency spectrum to file.

        :param str fid: String with file name to write to or an open file object.
        :param int precision: Precision with which to write out entries of the SFS.
            (They  are formated via %.<p>g, where <p> is the precision.)
        :param list comment_lines: List of strings to be used as comment lines in
            the header of the output file.
        :param bool foldmaskinfo: If False, folding and mask and population label
                      information will not be saved.
        """
        # Open the file object.
        newfile = False
        if not hasattr(fid, "write"):
            newfile = True
            fid = open(fid, "w")

        # Write comments
        for line in comment_lines:
            fid.write("# ")
            fid.write(line.strip())
            fid.write(os.linesep)

        # Write out the shape of the fs
        fid.write("{0} ".format(self.sample_size))

        if foldmaskinfo:
            if not self.folded:
                fid.write("unfolded ")
            else:
                fid.write("folded ")

        fid.write(os.linesep)

        # Write the data to the file
        self.data.tofile(fid, " ", "%%.%ig" % precision)
        fid.write(os.linesep)

        if foldmaskinfo:
            # Write the mask to the file
            np.asarray(self.mask, int).tofile(fid, " ")
            fid.write(os.linesep)

        # Close file
        if newfile:
            fid.close()

    def fold(self):
        """
        Fold the two-locus spectrum by minor allele frequencies.
        """
        if self.folded:
            raise ValueError("Input Spectrum is already folded.")
        ns = self.shape[0] - 1
        folded = 0 * self
        for ii in range(ns + 1):
            for jj in range(ns + 1):
                for kk in range(ns + 1):
                    if self.mask[ii, jj, kk]:
                        continue
                    p = ii + jj
                    q = ii + kk
                    if p > ns / 2 and q > ns / 2:
                        # Switch A/a and B/b, so AB becomes ab, Ab becomes aB, etc
                        folded[ns - ii - jj - kk, kk, jj] += self.data[ii, jj, kk]
                        folded.mask[ii, jj, kk] = True
                    elif p > ns / 2:
                        # Switch A/a, so AB -> aB, Ab -> ab, aB -> AB, and ab -> Ab
                        folded[kk, ns - ii - jj - kk, ii] += self.data[ii, jj, kk]
                        folded.mask[ii, jj, kk] = True
                    elif q > ns / 2:
                        # Switch B/b, so AB -> Ab, Ab -> AB, aB -> ab, and ab -> aB
                        folded[jj, ii, ns - ii - jj - kk] += self.data[ii, jj, kk]
                        folded.mask[ii, jj, kk] = True
                    else:
                        folded[ii, jj, kk] += self.data[ii, jj, kk]
        folded.folded = True
        return folded

    def project(self, ns, finite_genome=False, cache=True):
        """
        Project to smaller sample size.

        param int ns: Sample size for new spectrum.
        param bool finite_genome: If we also track proportions in fixed bins.
        """
        if finite_genome:
            raise ValueError("Projection with finite genome is not currently supported")
        data = Numerics.project(self, ns, cache=cache)
        output = TLSpectrum(data, mask_infeasible=True)
        return output

    # Ensures that when arithmetic is done with TLSpectrum objects,
    # attributes are preserved. For details, see similar code in
    # moments.Spectrum_mod
    for method in [
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__div__",
        "__rdiv__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__rpow__",
        "__pow__",
    ]:
        exec(
            """
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, np.ma.masked_array):
        newdata = self.data.%(method)s (other.data)
        newmask = np.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.data.%(method)s (other)
        newmask = self.mask
    outfs = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   mask_infeasible=False, 
                                   data_folded=self.folded)
    return outfs
"""
            % {"method": method}
        )

    # Methods that modify the Spectrum in-place.
    for method in [
        "__iadd__",
        "__isub__",
        "__imul__",
        "__idiv__",
        "__itruediv__",
        "__ifloordiv__",
        "__ipow__",
    ]:
        exec(
            """
def %(method)s(self, other):
    self._check_other_folding(other)
    if isinstance(other, np.ma.masked_array):
        self.data.%(method)s (other.data)
        self.mask = np.ma.mask_or(self.mask, other.mask)
    else:
        self.data.%(method)s (other)
    return self
"""
            % {"method": method}
        )

    def _check_other_folding(self, other):
        """
        Ensure other Spectrum has same .folded status
        """
        if isinstance(other, self.__class__) and (other.folded != self.folded):
            raise ValueError(
                "Cannot operate with a folded Spectrum and an " "unfolded one."
            )

    def integrate(
        self,
        nu,
        tf,
        dt=0.01,
        rho=None,
        gamma=None,
        sel_params=None,
        sel_params_general=None,
        theta=1.0,
        finite_genome=False,
        u=None,
        v=None,
        alternate_fg=None,
        clustered_mutations=False,
    ):
        """
        Simulate the two-locus haplotype frequency spectrum forward in time.
        This integration scheme takes advantage of scipy's sparse methods.

        When using the reversible mutation model (with `finite_genome` = True),
        we are limited to selection at only one locus (the left locus), and
        selection is additive. When using the default ISM, additive selection
        is allowed at both loci, and we use `sel_params`, which specifies [sAB,
        sA, and sB] in that order.  Note that while this selection model is
        additive within loci, it allows for epistasis between loci if sAB != sA
        + sB.

        :param nu: Population effective size as positive value or callable function.
        :param float tf: The integration time in genetics units.
        :param float dt_fac: The time step for integration.
        :param float rho: The population-size scaled recombination rate 4*Ne*r.
        :param float gamma: The population-size scaled selection coefficient 2*Ne*s.
        :param list sel_params: A list of selection parameters. See docstrings in
            Numerics. Selection parameters will be deprecated when we clean up the
            numerics and integration.
        :param list sel_params_general: To be filled. ## TODO!!
        :param float theta: Population size scale mutation parameter.
        :param bool finite_genome: Defaults to False, in which case we use the
            infinite sites model. Otherwise, we use a reversible mutation model, and
            must specify ``u`` and ``v``.
        :param float u: The mutation rate at the left locus in the finite genome model.
        :param float v: The mutation rate at the right locus in the finite genome
            model.
        :param bool alternate_fg: If True, use the alternative finite genome model.
            This parameter will be deprecated when we clean up the numerics and
            integration.
        """
        if gamma == None:
            gamma = 0.0
        if rho == None:
            rho = 0.0
            print("Warning: rho was not specified. Simulating with rho = 0")

        self.data[:] = Integration.integrate(
            self.data,
            nu,
            tf,
            rho=rho,
            dt=dt,
            theta=theta,
            gamma=gamma,
            sel_params=sel_params,
            sel_params_general=sel_params_general,
            finite_genome=finite_genome,
            u=u,
            v=v,
            alternate_fg=alternate_fg,
            clustered_mutations=clustered_mutations,
        )


# Allow TLSpectrum objects to be pickled.
# See http://effbot.org/librarybook/copy-reg.htm
try:
    import copy_reg
except:
    import copyreg


def TLSpectrum_pickler(fs):
    # Collect all the info necessary to save the state of a TLSpectrum
    return TLSpectrum_unpickler, (fs.data, fs.mask, fs.folded)


def TLSpectrum_unpickler(data, mask, folded):
    # Use that info to recreate the TLSpectrum
    return TLSpectrum(data, mask, mask_infeasible=False, data_folded=folded)


try:
    copy_reg.pickle(TLSpectrum, TLSpectrum_pickler, TLSpectrum_unpickler)
except:
    copyreg.pickle(TLSpectrum, TLSpectrum_pickler, TLSpectrum_unpickler)
