"""
Contains Spectrum object, which represents frequency spectra.
"""

import logging

logging.basicConfig()
logger = logging.getLogger("Spectrum_mod")
import functools
import operator
import os
import sys
import numpy as np
from numpy import newaxis as nuax
import scipy.stats
import copy
import itertools
import warnings
import demes

import moments.Integration
import moments.Integration_nomig
from . import Numerics
from . import Parsing
import moments.Demes.Demes

_plotting = True
try:
    import moments.ModelPlot
except ImportError:
    # if matplotlib is not present, do not import, and do not run plotting functions
    _plotting = False


class Spectrum(np.ma.masked_array):
    """
    Represents a single-locus biallelic frequency spectrum.

    Spectra are represented by masked arrays. The masking allows us to ignore
    specific entries in the spectrum. When simulating under the standard infinite
    sites model (ISM), the entries we mask are the bins specifying absent or fixed
    variants. When using a reversible mutation model (i.e. the finite genome model),
    we track the density of variants in fixed bins, setting ``mask_corners`` to
    ``False``.

    :param data: An array with dimension equal to the number of populations.
        Each dimension has length :math:`n_i+1`, where :math:`n_i` is the
        sample size for the i-th population.
    :type data: array
    :param mask: An optional array of the same size as data. 'True' entries in
        this array are masked in the Spectrum. These represent missing
        data categories. (For example, you may not trust your singleton
        SNP calling.)
    :type maks: array, optional
    :param mask_corners: If True (default), the 'observed in none' and 'observed
        in all' entries of the FS will be masked. Typically these
        entries are masked. In the defaul infinite sites model, moments does
        not reliably calculate the fixed-bin entries, so you will almost always
        want ``mask_corners=True``. The exception is if we are simulating under
        the finite genome model, in which case we track the probability of
        a site to be fixed for either allele.
    :type maks_corners: bool, optional
    :param data_folded: If True, it is assumed that the input data is folded. An
        error will be raised if the input data and mask are not
        consistent with a folded Spectrum.
    :type data_folded: bool, optional
    :param check_folding: If True and data_folded=True, the data and mask will be
        checked to ensure they are consistent with a folded
        Spectrum. If they are not, a warning will be printed.
    :type check_folding: bool, optional
    :param pop_ids: Optional list of strings containing the population labels,
        with length equal to the dimension of ``data``.
    :type pop_ids: list of strings, optional

    :return: A frequency spectrum object, as a masked array.
    """

    def __new__(
        subtype,
        data,
        mask=np.ma.nomask,
        mask_corners=True,
        data_folded=None,
        check_folding=True,
        dtype=float,
        copy=True,
        fill_value=np.nan,
        keep_mask=True,
        shrink=True,
        pop_ids=None,
    ):
        data = np.asanyarray(data)

        if mask is np.ma.nomask:
            mask = np.ma.make_mask_none(data.shape)
        else:
            # Since a mask is given, we do not mask the corners
            mask_corners = False

        subarr = np.ma.masked_array(
            data,
            mask=mask,
            dtype=dtype,
            copy=copy,
            fill_value=np.nan,
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

        # Check that if we're declaring that the input data is folded, it
        # actually is, and the mask reflects this.
        if data_folded:
            total_samples = np.sum(subarr.sample_sizes)
            total_per_entry = subarr._total_per_entry()
            # Which entries are nonsense in the folded fs.
            where_folded_out = total_per_entry > int(total_samples / 2)
            if check_folding and not np.all(subarr.data[where_folded_out] == 0):
                logger.warn(
                    "Creating Spectrum with data_folded = True, but "
                    "data has non-zero values in entries which are "
                    "nonsensical for a folded Spectrum."
                )
            if check_folding and not np.all(subarr.mask[where_folded_out]):
                logger.warn(
                    "Creating Spectrum with data_folded = True, but "
                    "mask is not True for all entries which are "
                    "nonsensical for a folded Spectrum."
                )

        if hasattr(data, "pop_ids"):
            if pop_ids is None or pop_ids == data.pop_ids:
                subarr.pop_ids = data.pop_ids
            elif pop_ids != data.pop_ids:
                logger.warn(
                    "Changing population labels in construction of new " "Spectrum."
                )
                if len(pop_ids) != subarr.ndim:
                    raise ValueError(
                        "pop_ids must be of length equal to "
                        "dimensionality of Spectrum."
                    )
                subarr.pop_ids = pop_ids
        else:
            if pop_ids is not None and len(pop_ids) != subarr.ndim:
                raise ValueError(
                    "pop_ids must be of length equal to " "dimensionality of Spectrum."
                )
            subarr.pop_ids = pop_ids

        if mask_corners is True:
            subarr.mask_corners()

        return subarr

    # See http://www.scipy.org/Subclasses for information on the
    # __array_finalize__ and __array_wrap__ methods. I had to do some debugging
    # myself to discover that I also needed _update_from.
    # Also, see http://docs.scipy.org/doc/np/reference/arrays.classes.html
    # Also, see http://docs.scipy.org/doc/np/user/basics.subclassing.html
    #
    # We need these methods to ensure extra attributes get copied along when
    # we do arithmetic on the FS.
    def __array_finalize__(self, obj):
        if obj is None:
            return
        np.ma.masked_array.__array_finalize__(self, obj)
        self.folded = getattr(obj, "folded", "unspecified")
        self.pop_ids = getattr(obj, "pop_ids", None)

    def __array_wrap__(self, obj, context=None, return_scalar=False):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, context, return_scalar)
        result.folded = self.folded
        result.pop_ids = self.pop_ids
        return result

    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, "folded"):
            self.folded = obj.folded
        if hasattr(obj, "pop_ids"):
            self.pop_ids = obj.pop_ids

    # masked_array has priority 20.
    __array_priority__ = 20

    def __repr__(self):
        return "Spectrum(%s, folded=%s, pop_ids=%s)" % (
            str(self),
            str(self.folded),
            str(self.pop_ids),
        )

    # Functions for manipulating frequency spectra.
    def mask_corners(self):
        """
        Mask the 'seen in 0 samples' and 'seen in all samples' entries.
        """
        self.mask.flat[0] = self.mask.flat[-1] = True

    def unmask_corners(self):
        """
        Unmask the 'seen in 0 samples' and 'seen in all samples' entries.
        """
        self.mask.flat[0] = self.mask.flat[-1] = False

    def unmask_all(self):
        """
        Unmask all entires of the frequency spectrum.
        """
        self.mask[tuple([slice(None)] * self.Npop)] = False

    def _get_sample_sizes(self):
        return np.asarray(self.shape) - 1

    sample_sizes = property(_get_sample_sizes)

    def _get_Npop(self):
        return self.ndim

    Npop = property(_get_Npop)

    def _ensure_dimension(self, Npop):
        """
        Ensure that fs has Npop dimensions.
        """
        if not self.Npop == Npop:
            raise ValueError("Only compatible with %id spectra." % Npop)

    def project(self, ns):
        """
        Project to smaller sample size.

        ``project`` does *not* act in-place, so that the input frequency
        spectrum is not changed.

        :param ns: Sample sizes for new spectrum.
        :type ns: list of integers
        """
        if len(ns) != self.Npop:
            raise ValueError(
                "Requested sample sizes not of same dimension "
                "as spectrum. Perhaps you need to marginalize "
                "over some populations first?"
            )
        if np.any(np.asarray(ns) > np.asarray(self.sample_sizes)):
            raise ValueError(
                "Cannot project to a sample size greater than "
                "original. Original size is %s and requested size "
                "is %s." % (self.sample_sizes, ns)
            )

        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # projecting.
        if original_folded:
            output = self.unfold()
        else:
            output = self.copy()

        # Iterate over each axis, applying the projection.
        for axis, proj in enumerate(ns):
            if proj != self.sample_sizes[axis]:
                output = output._project_one_axis(proj, axis)

        output.pop_ids = self.pop_ids

        # Return folded or unfolded as original.
        if original_folded:
            return output.fold()
        else:
            return output

    def _project_one_axis(self, n, axis=0):
        """
        Project along a single axis.
        """
        # This gets a little tricky with fancy indexing to make it work
        # for fs with arbitrary number of dimensions.
        if n > self.sample_sizes[axis]:
            raise ValueError(
                "Cannot project to a sample size greater than "
                "original. Called sizes were from %s to %s."
                % (self.sample_sizes[axis], n)
            )

        newshape = list(self.shape)
        newshape[axis] = n + 1
        # Create a new empty fs that we'll fill in below.
        pfs = Spectrum(np.zeros(newshape), mask_corners=False)

        # Set up for our fancy indexes. These slices are currently like
        # [:,:,...]
        from_slice = [slice(None) for ii in range(self.Npop)]
        to_slice = [slice(None) for ii in range(self.Npop)]
        proj_slice = [nuax for ii in range(self.Npop)]

        proj_from = self.sample_sizes[axis]
        # For each possible number of hits.
        for hits in range(proj_from + 1):
            # Adjust the slice in the array we're projecting from.
            from_slice[axis] = slice(hits, hits + 1)
            # These are the least and most possible hits we could have in the
            #  projected fs.
            least, most = max(n - (proj_from - hits), 0), min(hits, n)
            to_slice[axis] = slice(least, most + 1)
            # The projection weights.
            proj = Numerics._cached_projection(n, proj_from, hits)
            proj_slice[axis] = slice(least, most + 1)
            # Do the multiplications
            pfs.data[tuple(to_slice)] += (
                self.data[tuple(from_slice)] * proj[tuple(proj_slice)]
            )
            pfs.mask[tuple(to_slice)] = np.logical_or(
                pfs.mask[tuple(to_slice)], self.mask[tuple(from_slice)]
            )

        return pfs

    def marginalize(self, over, mask_corners=None):
        """
        Reduced dimensionality spectrum summing over the set of populations
        given by ``over``.

        ``marginalize`` does not act in-place, so the input frequency spectrum
        will not be altered.

        :param over: List of axes to sum over. For example (0,2) will marginalize
            populations 0 and 2.
        :type over: list of integers
        :param mask_corners: If True, the fixed bins of the resulting spectrum will be
            masked. The default behavior is to mask the corners only if at least one
            of the corners of the input frequency spectrum is masked. If either
            corner is masked, the output frequency spectrum masks the fixed bins.
        :type mask_corners: bool, optional
        """
        if _plotting:
            # Update ModelPlot
            model = moments.ModelPlot._get_model()
            if model is not None:
                model.extinction(over)

        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # marginalizing.
        if original_folded:
            output = self.unfold()
        else:
            output = self.copy()

        orig_mask = output.mask.copy()
        orig_mask.flat[0] = orig_mask.flat[-1] = False
        if np.any(orig_mask):
            logger.warn(
                "Marginalizing a Spectrum with internal masked values. "
                "This may not be a well-defined operation."
            )

        # Do the marginalization
        for axis in sorted(over)[::-1]:
            output = output.sum(axis=axis)
        pop_ids = None
        if self.pop_ids is not None:
            pop_ids = list(self.pop_ids)
            for axis in sorted(over)[::-1]:
                del pop_ids[axis]
        output.folded = False
        output.pop_ids = pop_ids

        if mask_corners is None:
            if self.mask.flat[0] == True or self.mask.flat[-1] == True:
                mask_corners = True
            else:
                mask_corners = False
        if mask_corners:
            output.mask_corners()

        # Return folded or unfolded as original.
        if original_folded:
            return output.fold()
        else:
            return output

    def swap_axes(self, ax1, ax2):
        """
        Uses np's swapaxes function, but also swaps pop_ids as appropriate
        if pop_ids are given.

        .. note::
            ``fs.swapaxes(ax1, ax2)`` will still work, but if population
            ids are given, it won't swap the pop_ids entries as expected.

        :param ax1: The index of the first population to swap.
        :type ax1: int
        :param ax2: The index of the second population to swap.
        :type ax2: int
        """
        output = np.swapaxes(self, ax1, ax2)
        if output.pop_ids is not None:
            pop1, pop2 = output.pop_ids[ax1], output.pop_ids[ax2]
            output.pop_ids[ax1], output.pop_ids[ax2] = pop2, pop1
        return output

    def _counts_per_entry(self):
        """
        Counts per population for each entry in the fs.
        """
        ind = np.indices(self.shape)
        # Transpose the first access to the last, so ind[ii,jj,kk] = [ii,jj,kk]
        ind = ind.transpose(list(range(1, self.Npop + 1)) + [0])
        return ind

    def _total_per_entry(self):
        """
        Total derived alleles for each entry in the fs.
        """
        return np.sum(self._counts_per_entry(), axis=-1)

    def log(self):
        """
        Returns the natural logarithm of the entries of the frequency spectrum.

        Only necessary because np.ma.log now fails to propagate extra
        attributes after np 1.10.
        """
        logfs = np.ma.log(self)
        logfs.folded = self.folded
        logfs.pop_ids = self.pop_ids
        return logfs

    def fold(self):
        """
        Returns a folded frequency spectrum.

        The folded fs assumes that information on which allele is ancestral or
        derived is unavailable. Thus the fs is in terms of minor allele
        frequency.  This makes the fs into a "triangular" array. If a masked cell
        is folded into non-masked cell, the destination cell is masked as well.

        Folding is not done in-place. The return value is a new Spectrum object.
        """
        if self.folded:
            raise ValueError("Input Spectrum is already folded.")

        # How many samples total do we have? The folded fs can only contain
        # entries up to total_samples/2 (rounded down).
        total_samples = np.sum(self.sample_sizes)

        total_per_entry = self._total_per_entry()

        # Here's where we calculate which entries are nonsense in the folded fs.
        where_folded_out = total_per_entry > int(total_samples / 2)

        original_mask = self.mask
        # Here we create a mask that masks any values that were masked in
        # the original fs (or folded onto by a masked value).
        final_mask = np.logical_or(original_mask, Numerics.reverse_array(original_mask))

        # To do the actual folding, we take those entries that would be folded
        # out, reverse the array along all axes, and add them back to the
        # original fs.
        reversed = Numerics.reverse_array(np.where(where_folded_out, self, 0))
        folded = np.ma.masked_array(self.data + reversed)
        folded.data[where_folded_out] = 0

        # Deal with those entries where assignment of the minor allele is
        # ambiguous.
        where_ambiguous = total_per_entry == total_samples / 2.0
        ambiguous = np.where(where_ambiguous, self, 0)
        folded += -0.5 * ambiguous + 0.5 * Numerics.reverse_array(ambiguous)

        # Mask out the remains of the folding operation.
        final_mask = np.logical_or(final_mask, where_folded_out)

        outfs = Spectrum(
            folded, mask=final_mask, data_folded=True, pop_ids=self.pop_ids
        )
        return outfs

    def unfold(self):
        """
        Returns an unfolded frequency spectrum.

        It is assumed that each state of a SNP is equally likely to be
        ancestral.

        Unfolding is not done in-place. The return value is a new Spectrum object.
        """
        if not self.folded:
            raise ValueError("Input Spectrum is not folded.")

        # Unfolding the data is easy.
        reversed_data = Numerics.reverse_array(self.data)
        newdata = (self.data + reversed_data) / 2.0

        # Unfolding the mask is trickier. We want to preserve masking of entries
        # that were masked in the original Spectrum.
        # Which entries in the original Spectrum were masked solely because
        # they are incompatible with a folded Spectrum?
        total_samples = np.sum(self.sample_sizes)
        total_per_entry = self._total_per_entry()
        where_folded_out = total_per_entry > int(total_samples / 2)

        newmask = np.logical_xor(self.mask, where_folded_out)
        newmask = np.logical_or(newmask, Numerics.reverse_array(newmask))

        outfs = Spectrum(newdata, mask=newmask, data_folded=False, pop_ids=self.pop_ids)
        return outfs

    # Functions that apply demographic events, including integration.
    def split(self, idx, n0, n1, new_ids=None):
        """
        Splits a population in the SFS into two populations, with the extra
        population placed at the end. Returns a new frequency spectrum.

        :param idx: The index of the population to split.
        :type idx: int
        :param n0: The sample size of the first split population.
        :type n0: int
        :param n1: The sample size of the second split population.
        :type n1: int
        :param new_ids: The population IDs of the split populations. Can only be
            used if pop_ids are given for the input spectrum.
        :type new_ids: list of strings, optional
        """
        if self.folded:
            raise ValueError("Cannot perform split on folded spectrum.")
        if self.pop_ids is None and new_ids is not None:
            raise ValueError("Trying to assign ids to a SFS with no pop_ids.")
        if new_ids is not None and len(new_ids) != 2:
            raise ValueError("new_ids must be a list of two population id strings")
        fs_split = moments.Manips.split_by_index(self, idx, n0, n1)
        if new_ids is not None:
            fs_split.pop_ids = self.pop_ids
            fs_split.pop_ids[idx] = new_ids[0]
            fs_split.pop_ids.append(new_ids[1])
        return fs_split

    def branch(self, idx, n, new_id=None):
        """
        A "branch" event, where a population gives rise to a child population, while
        persisting. This is conceptually similar to the split event. The number of
        lineages in the new population is provided, and the number of lineages in the
        source/parental population is the original sample size minus the number
        requested for the branched population. Returns a new frequency spectrum.

        :param idx: The index of the population to branch.
        :type idx: int
        :param n: The sample size of the new population.
        :type n: int
        :param new_id: The population ID of the branch populations. The parental
            population retains its original population ID. Can only be
            used if pop_ids are given for the input spectrum.
        :type new_ids: str, optional
        """
        if self.folded:
            raise ValueError("Cannot perform branch on folded spectrum.")
        if self.pop_ids is None and new_id is not None:
            raise ValueError("Trying to assign ids to a SFS with no pop_ids.")
        if idx < 0 or idx >= self.Npop:
            raise ValueError(
                f"Cannot branch population index {idx} in SFS of size {self.Npop}"
            )
        n_parent = self.sample_sizes[idx]
        n0 = n_parent - n
        fs_branch = moments.Manips.split_by_index(self, idx, n0, n)
        if new_id is not None:
            fs_branch.pop_ids = self.pop_ids
            fs_branch.pop_ids.append(new_id)
        return fs_branch

    def admix(self, idx0, idx1, num_lineages, proportion, new_id=None):
        """
        Returns a new frequency spectrum with an admixed population that arose through
        admixture from indexed populations with given number of lineages and
        proportions from parental populations. This serves as a wrapper for
        ``Manips.admix_into_new``, with the added feature of handling pop_ids.

        If the number of lineages that move are equal to the number
        of lineages previously present in a source population, that source
        population is marginalized.

        :param idx0: Index of first source population.
        :type idx0: int
        :param idx1: Index of second source population.
        :type idx1: int
        :param num_lineages: Number of lineages in the new population. Cannot be
            greater than the number of existing lineages in either source
            populations.
        :type num_lineages: int
        :param proportion: The proportion of lineages that come from the first
            source population (1-proportion acestry comes from the second source
            population). Must be a number between 0 and 1.
        :type proportion: float
        :param new_id: The ID of the new population. Can only be used if the
            population IDs are specified in the input SFS.
        :type new_id: str, optional
        """
        if new_id is not None and self.pop_ids is None:
            raise ValueError("Cannot specify new pop ids if input SFS has no pop_ids")
        if proportion < 0 or proportion > 1:
            raise ValueError("proportion must be between 0 and 1")
        if idx0 == idx1:
            raise ValueError("Cannot admix population with itself")
        if idx0 < 0 or idx0 >= self.Npop or idx1 < 0 or idx1 >= self.Npop:
            raise ValueError(f"Population indexes must be between 0 and {self.Npop-1}")
        fs_admix = moments.Manips.admix_into_new(
            self, idx0, idx1, num_lineages, proportion
        )
        if new_id is not None:
            new_pop_ids = copy.copy(self.pop_ids)
            # remove pop ids for marginalized pops
            for idx in sorted([idx0, idx1])[::-1]:
                if self.sample_sizes[idx] == num_lineages:
                    del new_pop_ids[idx]
            new_pop_ids.append(new_id)
            fs_admix.pop_ids = new_pop_ids
        return fs_admix

    def pulse_migrate(self, idx_from, idx_to, keep_from, proportion):
        """
        Mass migration (pulse admixture) between two existing populations. The
        target (destination) population has the same number of lineages in the
        output SFS, and the source population has ``keep_from`` number of lineages
        after the pulse event. The proportion is the expected ancestry proportion
        in the target population that comes from the source population.

        This serves as a wrapper for ``Manips.admix_inplace``.

        Depending on the proportion and number of lineages, because this is an
        approximate operation, we often need a large number of lineages from
        the source population to maintain accuracy.

        :param idx_from: Index of source population.
        :type idx_from: int
        :param idx_to: Index of targeet population.
        :type idx_to: int
        :param keep_from: Number of lineages to keep in source population.
        :type keep_from: int
        :param proportion: Ancestry proportion of source population that migrates
            to target population.
        :type proportion: float
        """
        if idx_from < 0 or idx_from >= self.Npop or idx_to < 0 or idx_to >= self.Npop:
            raise ValueError(f"Invalid population index for {self.Npop}D SFS.")
        if proportion < 0 or proportion > 1:
            raise ValueError("proportion must be between 0 and 1")
        if idx_from == idx_to:
            raise ValueError("Cannot admix population into itself")
        fs_pulse = moments.Manips.admix_inplace(
            self, idx_from, idx_to, keep_from, proportion
        )
        if self.pop_ids is not None:
            fs_pulse.pop_ids = self.pop_ids
        return fs_pulse

    # Integrate the SFS in-place
    def integrate(
        self,
        Npop,
        tf,
        dt_fac=0.02,
        gamma=None,
        h=None,
        overdominance=None,
        m=None,
        theta=1.0,
        adapt_dt=False,
        finite_genome=False,
        theta_fd=None,
        theta_bd=None,
        frozen=[False],
    ):
        """
        Method to simulate the spectrum's evolution for a given set of demographic
        parameters. The SFS is integrated forward-in-time, and the integration
        occurs in-place, meaning you need only call ``fs.integrate( )``, and the
        ``fs`` is updated.

        :param Npop: List of populations' relative effective sizes. Can be given
            as a list of positive values for constant sizes, or as a function that
            returns a list of sizes at a given time.
        :type Npop: list or function that returns a list
        :param tf: The total integration time in genetic units.
        :type tf: float
        :param dt_fac: The timestep factor, default is 0.02. This parameter typically
            does not need to be adjusted.
        :type dt_fac: float, optional
        :param gamma: Scaled selection coefficient(s), which can be either a number
            or a vector gamma = [gamma1,...,gammap], allowing for different selection
            coefficients in each population. This can also be given as function
            returning a number or such a vector, allowing for selection coefficients
            to change over time.
        :type gamma: float or list of floats or function that returns a float or
            list of floats, optional
        :param h: Dominance coefficient(s) (h=1/2 implies additive selection). Can be
            given as a number or a vector h = [h1,...,hp], or a function that returns
            a number or vector of coefficients, allowing for dominance coefficients
            to change over time.
        :type h: float or list of floats or function that returns a float or
            list of floats, optional
        :param overdominance: Scaled selection coefficient(s) that is applied only to
            heterozygotes, in a selection system with fitnesses 1:1+s:1. Underdominance
            can be modeled by passing a negative value. Not that this is a symmetric
            under/over-dominance model, in which homozygotes for either the ancestral
            or derived allele have equal fitness. ``gamma``, ``h``, and ``overdominance``
            can be combined (additively) to implement non-symmetric selection
            scenarios.
        :type overdominance: float or list of floats or function that returns a float or
            list of floats, optional
        :param m: The migration rates matrix as a 2-D array with shape nxn,
            where n is the number of populations. The entry of the migration
            matrix m[i,j] is the migration rate from pop j to pop i in genetic
            units, that is, normalized by :math:`2N_e`. ``m`` may be either a
            2-D array, or a function that returns a 2-D array (with dimensions
            equal to (num pops)x(num pops)).
        :type m: array-like, optional
        :param theta: The scaled mutation rate :math:`4 N_e u`, which defaults to 1.
            ``theta`` can be used in the reversible model in the case of symmetric
            mutation rates. In this case, ``theta`` must be set to << 1.
        :type theta: float, optional
        :param adapt_dt: flag to allow dt correction avoiding negative entries.
        :type adapt_dt: bool, optional
        :param finite_genome: If True, simulate under the finite-genome model with
            reversible mutations. If using this model, we can specify the forward
            and backward mutation rates, which are per-base rates that are not
            scaled by number of mutable loci. If ``theta_fd`` and ``theta_bd``
            are not specified, we assume equal forward and backward mutation rates
            provided by ``theta``, which must be set to less that 1.
            Defaults to False.
        :type finite_genome: bool, optional
        :param theta_fd: The forward mutation rate :math:`4 Ne u`.
        :type theta_fd: float, optional
        :param theta_bd: The backward mutation rate :math:`4 Ne v`.
        :type theta_bd: float, optional
        :param frozen: Specifies the populations that are "frozen", meaning
            samples from that population no longer change due or contribute
            to migration to other populations. This feature is most often
            used to indicate ancient samples, for example, ancient DNA.
            The ``frozen`` parameter is given as a list of same length
            as number of pops, with ``True`` for frozen
            populations at the corresponding index, and ``False`` for
            populations that continue to evolve.
        :type frozen: list of bools
        """

        if tf < 0:
            raise ValueError("Integration time cannot be negative")

        n = np.array(self.shape) - 1

        if m is not None and not callable(m):
            m = np.array(m)

        if finite_genome == True and (theta_fd == None or theta_bd == None):
            if theta >= 1:
                raise ValueError(
                    "In the finite genome model, theta must be much less than 1. "
                    "If symmetric mutation rates, can use theta << 1. Otherwise, "
                    "theta_fd and theta_bd must be specified."
                )
            else:
                theta_fd = theta_bd = theta

        if np.any(frozen):
            if (hasattr(Npop, "__len__") and len(Npop) != len(frozen)) or (
                callable(Npop) and len(Npop(0)) != len(frozen)
            ):
                raise ValueError(
                    "If one or more populations are frozen, length "
                    "of frozen must match number of simulated pops."
                )

        if _plotting:
            model = moments.ModelPlot._get_model()
            if model is not None:
                model.evolve(tf, Npop, m)

        if gamma is None:
            gamma = 0
        if h is None:
            h = 0.5
        if overdominance is None:
            overdominance = 0

        if len(n) == 1:
            if gamma == 0 and overdominance == 0:
                self.data[:] = moments.Integration_nomig.integrate_neutral(
                    self.data,
                    Npop,
                    tf,
                    dt_fac=dt_fac,
                    theta=theta,
                    finite_genome=finite_genome,
                    theta_fd=theta_fd,
                    theta_bd=theta_bd,
                    frozen=frozen,
                )
            else:
                self.data[:] = moments.Integration_nomig.integrate_nomig(
                    self.data,
                    Npop,
                    tf,
                    dt_fac=dt_fac,
                    gamma=gamma,
                    h=h,
                    overdominance=overdominance,
                    theta=theta,
                    finite_genome=finite_genome,
                    theta_fd=theta_fd,
                    theta_bd=theta_bd,
                    frozen=frozen,
                )
        else:
            if m is None:
                m = np.zeros([len(n), len(n)])
            if not callable(m) and (m == 0).all():
                # for more than 2 populations, the sparse solver
                # seems to be faster than the tridiag...
                if (
                    (np.array(gamma) == 0).all()
                    and (np.array(overdominance) == 0).all()
                    and len(n) < 3
                ):
                    self.data[:] = moments.Integration_nomig.integrate_neutral(
                        self.data,
                        Npop,
                        tf,
                        dt_fac=dt_fac,
                        theta=theta,
                        finite_genome=finite_genome,
                        theta_fd=theta_fd,
                        theta_bd=theta_bd,
                        frozen=frozen,
                    )
                else:
                    self.data[:] = moments.Integration_nomig.integrate_nomig(
                        self.data,
                        Npop,
                        tf,
                        dt_fac=dt_fac,
                        gamma=gamma,
                        h=h,
                        overdominance=overdominance,
                        theta=theta,
                        finite_genome=finite_genome,
                        theta_fd=theta_fd,
                        theta_bd=theta_bd,
                        frozen=frozen,
                    )
            else:
                self.data[:] = moments.Integration.integrate_nD(
                    self.data,
                    Npop,
                    tf,
                    dt_fac=dt_fac,
                    gamma=gamma,
                    h=h,
                    overdominance=overdominance,
                    m=m,
                    theta=theta,
                    adapt_dt=adapt_dt,
                    finite_genome=finite_genome,
                    theta_fd=theta_fd,
                    theta_bd=theta_bd,
                    frozen=frozen,
                )

    # Functions for computing statistics from frequency spetra.

    def Fst(self, pairwise=False):
        """
        Wright's Fst between the populations represented in the fs.

        This estimate of Fst assumes random mating, because we don't have
        heterozygote frequencies in the fs.

        Calculation is by the method of Weir and Cockerham *Evolution* 38:1358
        (1984). For a single SNP, the relevant formula is at the top of page
        1363. To combine results between SNPs, we use the weighted average
        indicated by equation 10.

        :param pairwise: Defaults to False. If True, returns a dictionary
            of all pairwise Fst within the multi-dimensional spectrum.
        :type pairwise: bool
        """
        if pairwise:
            Fsts = {}
            for pair in itertools.combinations(range(self.ndim), 2):
                to_marginalize = list(range(self.ndim))
                to_marginalize.pop(to_marginalize.index(pair[0]))
                to_marginalize.pop(to_marginalize.index(pair[1]))
                fs_marg = self.marginalize(to_marginalize)
                if self.pop_ids is not None:
                    pair_key = (self.pop_ids[pair[0]], self.pop_ids[pair[1]])
                else:
                    pair_key = pair
                Fsts[pair_key] = fs_marg.Fst()
            return Fsts
        else:
            # This gets a little obscure because we want to be able to work with
            # spectra of arbitrary dimension.

            # First quantities from page 1360
            r = self.Npop
            ns = self.sample_sizes
            nbar = np.mean(ns)
            nsum = np.sum(ns)
            nc = (nsum - np.sum(ns**2) / nsum) / (r - 1)

            # counts_per_pop is an r+1 dimensional array, where the last axis simply
            # records the indices of the entry.
            # For example, counts_per_pop[4,19,8] = [4,19,8]
            counts_per_pop = np.indices(self.shape)
            counts_per_pop = np.transpose(
                counts_per_pop, axes=list(range(1, r + 1)) + [0]
            )

            # The last axis of ptwiddle is now the relative frequency of SNPs in
            # that bin in each of the populations.
            ptwiddle = 1.0 * counts_per_pop / ns

            # Note that pbar is of the same shape as fs...
            pbar = np.sum(ns * ptwiddle, axis=-1) / nsum

            # We need to use 'this_slice' to get the proper aligment between
            # ptwiddle and pbar.
            this_slice = [slice(None)] * r + [np.newaxis]
            s2 = np.sum(ns * (ptwiddle - pbar[tuple(this_slice)]) ** 2, axis=-1) / (
                (r - 1) * nbar
            )

            # Note that this 'a' differs from equation 2, because we've used
            # equation 3 and b = 0 to solve for hbar.
            a = (
                nbar
                / nc
                * (s2 - 1 / (2 * nbar - 1) * (pbar * (1 - pbar) - (r - 1) / r * s2))
            )
            d = 2 * nbar / (2 * nbar - 1) * (pbar * (1 - pbar) - (r - 1) / r * s2)

            # The weighted sum over loci.
            asum = (self * a).sum()
            dsum = (self * d).sum()

            return asum / (asum + dsum)

    def S(self):
        """
        Returns the number of segregating sites in the frequency spectrum.
        """
        oldmask = self.mask.copy()
        self.mask_corners()
        S = self.sum()
        self.mask = oldmask
        return S

    def Watterson_theta(self):
        """
        Returns Watterson's estimator of theta.

        .. note::
            This function is only sensible for 1-dimensional spectra.
        """
        if self.Npop != 1:
            raise ValueError("Only defined on a one-dimensional fs.")

        n = self.sample_sizes[0]
        S = self.S()
        an = np.sum(1.0 / np.arange(1, n))

        return S / an

    def theta_L(self):
        """
        Returns theta_L as defined by Zeng et al. "Statistical Tests for Detecting
        Positive Selection by Utilizing High-Frequency Variants" (2006)
        Genetics

        .. note::
            This function is only sensible for 1-dimensional spectra.
        """
        if self.Npop != 1:
            raise ValueError("Only defined on a one-dimensional fs.")

        n = self.sample_sizes[0]
        return np.sum(np.arange(1, n) * self[1:n]) / (n - 1)

    def Zengs_E(self):
        """
        Returns Zeng et al.'s E statistic.

        From Zeng et al., "Statistical Tests for Detecting Positive Selection by
        Utilizing High-Frequency Variants." *Genetics*, 2016.
        """
        num = self.theta_L() - self.Watterson_theta()

        n = self.sample_sizes[0]

        # See after Eq. 3
        an = np.sum(1.0 / np.arange(1, n))
        # See after Eq. 9
        bn = np.sum(1.0 / np.arange(1, n) ** 2)
        s = self.S()

        # See immediately after Eq. 12
        theta = self.Watterson_theta()
        theta_sq = s * (s - 1.0) / (an**2 + bn)

        # Eq. 14
        var = (n / (2.0 * (n - 1.0)) - 1.0 / an) * theta + (
            bn / an**2
            + 2.0 * (n / (n - 1.0)) ** 2 * bn
            - 2 * (n * bn - n + 1.0) / ((n - 1.0) * an)
            - (3.0 * n + 1.0) / (n - 1.0)
        ) * theta_sq

        return num / np.sqrt(var)

    def pi(self):
        """
        Returns the estimated expected number of pairwise differences between two
        chromosomes in the population.

        .. note::
            This estimate includes a factor of sample_size / (sample_size - 1)
            to make :math:`\\mathbb{E}[\\pi] = \\theta`.
        """
        if self.ndim != 1:
            raise ValueError("Only defined for a one-dimensional SFS.")

        n = self.sample_sizes[0]
        # sample frequencies p
        p = np.arange(0, n + 1, dtype=float) / n
        # This expression derives from Gillespie's _Population_Genetics:_A
        # _Concise_Guide_, 2nd edition, section 2.6.
        return n / (n - 1.0) * 2 * np.ma.sum(self * p * (1 - p))

    def Tajima_D(self):
        """
        Returns Tajima's D.

        Following Gillespie "Population Genetics: A Concise Guide" pg. 45

        """
        if not self.Npop == 1:
            raise ValueError("Only defined on a one-dimensional SFS.")

        S = self.S()

        n = 1.0 * self.sample_sizes[0]
        pihat = self.pi()
        theta = self.Watterson_theta()

        a1 = np.sum(1.0 / np.arange(1, n))
        a2 = np.sum(1.0 / np.arange(1, n) ** 2)
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - 1.0 / a1
        c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2

        C = np.sqrt((c1 / a1) * S + c2 / (a1**2 + a2) * S * (S - 1))

        return (pihat - theta) / C

    def f2(self, X, Y):
        """
        Returns :math:`f_2(X, Y) = (pX-pY)^2`, where :math:`p` are observed
        allele frequencies in populations X and Y.

        X, and Y can be specified as population ID strings, or as indexes
        (but these cannot be mixed).

        :param X: One of the populations, as index or population ID.
        :param Y: The other population, as index or population ID.
        :return: Patterson's f2 statistics.
        """
        if type(X) is int and type(Y) is int:
            for _ in [X, Y]:
                if _ < 0 or _ >= self.Npop:
                    raise ValueError(
                        f"Population index {_} out of bounds for SFS "
                        f"with {self.Npop} populations"
                    )
            idx0 = X
            idx1 = Y
        elif type(X) is str and type(Y) is str:
            if self.pop_ids is None:
                raise ValueError("Populations given a strings, but pop_ids is None")
            for _ in [X, Y]:
                if _ not in self.pop_ids:
                    raise ValueError(f"Population {_} is not in pop_ids")
            idx0 = self.pop_ids.index(X)
            idx1 = self.pop_ids.index(Y)
        else:
            raise ValueError(
                "Populations X and Y must both be population indexes or IDs"
            )

        if idx0 == idx1:
            return 0

        to_marginalize = [i for i in range(self.Npop) if i not in [idx0, idx1]]
        fs = self.marginalize(to_marginalize)

        # symmetric statistic, so idx0 vs idx1 doesn't matter
        n0 = fs.sample_sizes[0]
        n1 = fs.sample_sizes[1]
        i0 = np.arange(n0 + 1)[:, np.newaxis]
        i1 = np.arange(n1 + 1)[np.newaxis, :]

        # To account for sampling bias
        fac = (
            i0 * (i0 - 1) / n0 / (n0 - 1)
            - 2 * i0 * i1 / n0 / n1
            + i1 * (i1 - 1) / n1 / (n1 - 1)
        )
        return np.sum(fac * fs)

    def f3(self, X, Y, Z):
        """
        Returns :math:`f_3(X; Y, Z) = (X-Y)(X-Z)`. A significantly negative
        :math:`f_3` suggests that population X is the result of admixture
        between ancient populations related to Y and Z. A positive value
        suggests that X is an outgroup to Y and Z.

        X, Y, and Z can be specified as population ID strings, or as indexes
        (but these cannot be mixed).

        :param X: The "test" population, as index or population ID.
        :param Y: The first reference population, as index or population ID.
        :param Z: The second reference population, as index or population ID.
        :returns: Patterson's f3 statistic.
        """
        if type(X) is int and type(Y) is int and type(Z) is int:
            for _ in [X, Y, Z]:
                if _ < 0 or _ >= self.Npop:
                    raise ValueError(
                        f"Population index {_} out of bounds for SFS "
                        f"with {self.Npop} populations"
                    )
            idx0 = X
            idx1 = Y
            idx2 = Z
        elif type(X) is str and type(Y) is str and type(Z) is str:
            if self.pop_ids is None:
                raise ValueError("Populations given a strings, but pop_ids is None")
            for _ in [X, Y, Z]:
                if _ not in self.pop_ids:
                    raise ValueError(f"Population {_} is not in pop_ids")
            idx0 = self.pop_ids.index(X)
            idx1 = self.pop_ids.index(Y)
            idx2 = self.pop_ids.index(Z)
        else:
            raise ValueError(
                "Populations X, Y and Z must each be population indexes or IDs"
            )

        if idx0 == idx1 or idx0 == idx2:
            return 0

        if idx1 == idx2:
            return self.f2(idx0, idx1)

        # reorder as needed so X is first population index
        fs = copy.deepcopy(self)
        if idx0 > idx1 or idx0 > idx2:
            fs = fs.swap_axes(0, idx0)
            if idx1 == 0:
                idx1 = idx0
            if idx2 == 0:
                idx2 = idx0
            idx0 = 0

        to_marginalize = [i for i in range(self.Npop) if i not in [idx0, idx1, idx2]]
        fs = fs.marginalize(to_marginalize)

        n0, n1, n2 = fs.sample_sizes
        i0 = np.arange(n0 + 1)[:, np.newaxis, np.newaxis]
        i1 = np.arange(n1 + 1)[np.newaxis, :, np.newaxis]
        i2 = np.arange(n2 + 1)[np.newaxis, np.newaxis, :]

        # To account for sampling bias
        fac = (
            i0 * (i0 - 1) / n0 / (n0 - 1)
            - i0 * i1 / n0 / n1
            - i0 * i2 / n0 / n2
            + i1 * i2 / n1 / n2
        )
        return np.sum(fac * fs)

    def f4(self, X, Y, Z, W):
        """
        Returns :math:`f_4(X, Y; Z, W) = (X-Y)(Z-W)`.

        X, Y, Z and W can be specified as population ID strings, or as indexes
        (but these cannot be mixed).

        :param X: A population index or ID.
        :param Y: A population index or ID.
        :param Z: A population index or ID.
        :param W: A population index or ID.
        :returns: Patterson's f4 statistic (pX-pY)*(pZ-pW).
        """
        if type(X) is int and type(Y) is int and type(Z) is int and type(W) is int:
            for _ in [X, Y, Z, W]:
                if _ < 0 or _ >= self.Npop:
                    raise ValueError(
                        f"Population index {_} out of bounds for SFS "
                        f"with {self.Npop} populations"
                    )
            idx0 = X
            idx1 = Y
            idx2 = Z
            idx3 = W
        elif type(X) is str and type(Y) is str and type(Z) is str and type(W) is str:
            if self.pop_ids is None:
                raise ValueError("Populations given a strings, but pop_ids is None")
            for _ in [X, Y, Z, W]:
                if _ not in self.pop_ids:
                    raise ValueError(f"Population {_} is not in pop_ids")
            idx0 = self.pop_ids.index(X)
            idx1 = self.pop_ids.index(Y)
            idx2 = self.pop_ids.index(Z)
            idx3 = self.pop_ids.index(W)
        else:
            raise ValueError(
                "Populations X, Y and Z must each be population indexes or IDs"
            )

        if X == Y or Z == W:
            return 0
        elif X == Z and Y == W:
            return self.f2(X, Z)
        elif X == W and Y == Z:
            return -self.f2(X, Z)
        elif X == Z:
            return self.f3(X, Y, W)
        elif X == W:
            return -self.f3(X, Y, Z)
        elif Y == Z:
            return -self.f3(Y, X, W)
        elif Y == W:
            return self.f3(Y, X, Z)
        else:
            assert len(set([X, Y, Z, W])) == 4

        # marginalize down to four populations
        to_marginalize = [
            i for i in range(self.Npop) if i not in [idx0, idx1, idx2, idx3]
        ]
        fs = self.marginalize(to_marginalize)

        # reorder as needed so that X, Y, Z, W are in order
        sort_indexes = np.argsort([idx0, idx1, idx2, idx3])
        for i in range(3):
            if sort_indexes[i] != i:
                assert sort_indexes[i] > i
                to_swap = np.argwhere(sort_indexes == i)[0][0]
                fs = fs.swap_axes(i, to_swap)
                sort_indexes[to_swap] = sort_indexes[i]
                sort_indexes[i] = i

        n0, n1, n2, n3 = fs.sample_sizes
        i0 = np.arange(n0 + 1)[:, np.newaxis, np.newaxis, np.newaxis]
        i1 = np.arange(n1 + 1)[np.newaxis, :, np.newaxis, np.newaxis]
        i2 = np.arange(n2 + 1)[np.newaxis, np.newaxis, :, np.newaxis]
        i3 = np.arange(n3 + 1)[np.newaxis, np.newaxis, np.newaxis, :]
        # To account for sampling bias
        fac = (
            i0 * i2 / n0 / n2
            - i0 * i3 / n0 / n3
            - i1 * i2 / n1 / n2
            + i1 * i3 / n1 / n3
        )
        return np.sum(fac * fs)

    # Functions for resampling or generating "data" from an SFS

    def fixed_size_sample(self, nsamples, include_masked=False):
        """
        Generate a resampled SFS from the current one. Thus, the resampled SFS
        follows a multinomial distribution given by the proportion of sites
        in each bin in the original SFS.

        :param nsamples: Number of samples to include in the new SFS.
        :type nsamples: int
        :param include_masked: If True, use all bins from the SFS. Otherwise,
            use only non-masked bins. Defaults to False.
        :type include_masked: bool, optional
        """
        flat = self.flatten()
        pvals = flat.data
        if include_masked is False:
            pvals[flat.mask] = 0
        pvals /= pvals.sum()

        sample = np.random.multinomial(int(nsamples), pvals)
        sample = sample.reshape(self.shape)
        sample = moments.Spectrum(sample, mask=self.mask, pop_ids=self.pop_ids)
        sample = sample.astype(int)

        return sample

    def sample(self):
        """
        Generate a Poisson-sampled fs from the current one.

        Entries where the current fs is masked or 0 will be masked in the
        output sampled fs.
        """
        # These are entries where the sampling has no meaning. Either the fs is
        # 0 there or masked.
        bad_entries = np.logical_or(self == 0, self.mask)
        # We convert to a 1-d array for passing into the sampler
        means = self.ravel().copy()
        # Filter out those bad entries.
        means[bad_entries.ravel()] = 1
        # Sample
        samp = scipy.stats.distributions.poisson.rvs(means, size=len(means))
        # Replace bad entries...
        samp[bad_entries.ravel()] = 0
        # Convert back to a properly shaped array
        samp = samp.reshape(self.shape)
        # Convert to a fs and mask the bad entries
        samp = Spectrum(
            samp, mask=bad_entries, data_folded=self.folded, pop_ids=self.pop_ids
        )
        samp = samp.astype(int)
        return samp

    def genotype_matrix(
        self, num_sites=None, sample_sizes=None, diploid_genotypes=False
    ):
        """
        Generate a genotype matrix of independent loci. For multi-population spectra,
        the individual columns are filled in the sample order as the populations
        in the SFS.

        .. note::
            Sites in the output genotype matrix are necessarily separated by
            infinite recombination. The SFS assumes all loci are segregating
            independently, so there is no linkage between them.

        Returns a genotype matrix of size number of sites by total sample size.

        :param num_sites: Defaults to None, in which case we take a poisson sample from
            the SFS. Otherwise, we take a fixed number of sites.
        :param sample_sizes: The sample size in each population, as a list with length
            of the number of dimension (populations) in the SFS.
        :diploid_genotypes: Defaults to False, in which case we return a haplotype
            matrix of size (num_sites x sum(sample_sizes)). If True, we return a
            diploid genotype matrix (filled with 0, 1, 2) of size
            (num_sites x sum(sample_sizes)/2).
        """
        # project to sample size as needed
        if sample_sizes is None:
            sample_sizes = self.sample_sizes
        proj = self.project(sample_sizes)
        if diploid_genotypes:
            for ss in proj.sample_sizes:
                if ss % 2 != 0:
                    raise ValueError(
                        "Requested diploid genotypes, but the haploid sample "
                        f"sizes ({sample_sizes}) are not all even."
                    )

        # get sampled SFS
        if num_sites is None:
            samp = proj.sample()
            num_sites = samp.sum()
        else:
            samp = proj.fixed_size_sample(num_sites)
        assert num_sites == samp.S()  # remove once tested

        # fill in genotype matrix
        G = np.zeros((num_sites, sum(proj.sample_sizes)), dtype=int)
        i = 0
        for idx in np.transpose(np.nonzero(samp)):
            for _ in range(samp[tuple(idx)]):
                g = []
                for j, n in zip(idx, sample_sizes):
                    g += list(np.random.permutation([1] * j + [0] * (n - j)))
                G[i] = g
                i += 1

        # collapse diploids if needed
        hap_sum = G.sum(axis=1)
        if diploid_genotypes:
            G = G[:, ::2] + G[:, 1::2]
        assert np.all(G.sum(axis=1) == hap_sum)  # remove once tested

        # shuffle genotype matrix
        np.random.shuffle(G)

        return G

    # Functions for saving and loading frequency spectra.

    @staticmethod
    def from_file(fid, mask_corners=None, return_comments=False):
        """
        Read frequency spectrum from file.

        See ``to_file`` for details on the file format.

        :param fid: string with file name to read from or an open file object.
        :type fid: string
        :param mask_corners: If True, mask the 'absent in all samples' and 'fixed in
            all samples' entries. If False, the corners are unmasked. If unspecified
            (left as None), the mask is unchanged from the saved file.
        :type mask_corners: bool, optional
        :param return_comments: If true, the return value is (fs, comments), where
            comments is a list of strings containing the comments
            from the file (without #'s).
        :type return_comments: bool, optional
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
        shape_spl = line.split()
        if "folded" not in shape_spl and "unfolded" not in shape_spl:
            # This case handles the old file format
            shape = tuple([int(d) for d in shape_spl])
            folded = False
            pop_ids = None
        else:
            # This case handles the new file format
            shape, next_ii = [int(shape_spl[0])], 1
            while shape_spl[next_ii] not in ["folded", "unfolded"]:
                shape.append(int(shape_spl[next_ii]))
                next_ii += 1
            folded = shape_spl[next_ii] == "folded"
            # Are there population labels in the file?
            if len(shape_spl) > next_ii + 1:
                pop_ids = line.split('"')[1::2]
            else:
                pop_ids = None

        data = np.fromstring(fid.readline().strip(), count=np.prod(shape), sep=" ")
        # fromfile returns a 1-d array. Reshape it to the proper form.
        data = data.reshape(*shape)

        maskline = fid.readline().strip()
        if not maskline:
            # The old file format didn't have a line for the mask
            mask = np.ma.nomask
        else:
            # This case handles the new file format
            mask = np.fromstring(maskline, count=np.prod(shape), sep=" ")
            if mask_corners is True:
                mask[0] = mask[-1] = 1
            elif mask_corners is False:
                mask[0] = mask[-1] = 0
            mask = mask.reshape(*shape)

        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        fs = Spectrum(data, mask=mask, data_folded=folded, pop_ids=pop_ids)

        if not return_comments:
            return fs
        else:
            return fs, comments

    fromfile = from_file

    def to_file(self, fid, precision=16, comment_lines=[], foldmaskinfo=True):
        """
        Write frequency spectrum to file.

        The file format is:

        - Any number of comment lines beginning with a '#'
        - A single line containing N integers giving the dimensions of the fs
          array. So this line would be '5 5 3' for an SFS that was 5x5x3.
          (That would be 4x4x2 *samples*.)
        - On the *same line*, the string 'folded' or 'unfolded' denoting the
          folding status of the array
        - On the *same line*, optional strings each containing the population
          labels in quotes separated by spaces, e.g. "pop 1" "pop 2"
        - A single line giving the array elements. The order of elements is
          e.g.: fs[0,0,0] fs[0,0,1] fs[0,0,2] ... fs[0,1,0] fs[0,1,1] ...
        - A single line giving the elements of the mask in the same order as
          the data line. '1' indicates masked, '0' indicates unmasked.

        :param fid: string with file name to write to or an open file object.
        :type fid: string
        :param precision: precision with which to write out entries of the SFS. (They
            are formated via %.<p>g, where <p> is the precision.) Defaults to 16.
        :type precision: int, optional
        :param comment_lines: list of strings to be used as comment lines in the header
            of the output file.
        :type comment_lines: list of strings, optional
        :param foldmaskinfo: If False, folding and mask and population label
            information will not be saved.
        :type foldmaskinfo: bool, optional
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
        for elem in self.data.shape:
            fid.write("%i " % elem)

        if foldmaskinfo:
            if not self.folded:
                fid.write("unfolded")
            else:
                fid.write("folded")
            if self.pop_ids is not None:
                for label in self.pop_ids:
                    fid.write(' "%s"' % label)

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

    ## Overide the (perhaps confusing) original np tofile method.
    tofile = to_file

    def scramble_pop_ids(self, mask_corners=True):
        """
        Spectrum corresponding to scrambling individuals among populations.

        This is useful for assessing how diverged populations are.
        Essentially, it pools all the individuals represented in the fs and
        generates new populations of random individuals (without replacement)
        from that pool. If this fs is significantly different from the
        original, that implies population structure.
        """
        original_folded = self.folded
        # If we started with an folded Spectrum, we need to unfold before
        # projecting.
        if original_folded:
            self = self.unfold()

        total_samp = np.sum(self.sample_sizes)

        # First generate a 1d sfs for the pooled population.
        combined = np.zeros(total_samp + 1)
        # For each entry in the fs, this is the total number of derived alleles
        total_per_entry = self._total_per_entry()
        # Sum up to generate the equivalent 1-d spectrum.
        for derived, counts in zip(total_per_entry.ravel(), self.ravel()):
            combined[derived] += counts

        # Now resample back into a n-d spectrum
        # For each entry, this is the counts per popuation.
        #  e.g. counts_per_entry[3,4,5] = [3,4,5]
        counts_per_entry = self._counts_per_entry()
        # Reshape it to be 1-d, so we can iterate over it easily.
        counts_per_entry = counts_per_entry.reshape(np.prod(self.shape), self.ndim)
        resamp = np.zeros(self.shape)
        for counts, derived in zip(counts_per_entry, total_per_entry.ravel()):
            # The probability here is
            # (t1 choose d1)*(t2 choose d2)/(ntot choose derived)
            lnprob = sum(
                Numerics._lncomb(t, d) for t, d in zip(self.sample_sizes, counts)
            )
            lnprob -= Numerics._lncomb(total_samp, derived)
            prob = np.exp(lnprob)
            # Assign result using the appropriate weighting
            resamp[tuple(counts)] += prob * combined[derived]

        resamp = Spectrum(resamp, mask_corners=mask_corners)
        if not original_folded:
            return resamp
        else:
            return resamp.fold()

    @staticmethod
    def from_data_dict(
        data_dict, pop_ids, projections, mask_corners=True, polarized=True
    ):
        """
        Spectrum from a dictionary of polymorphisms.

        The data dictionary should be organized as:

        .. code-block::

            {snp_id: {
                'segregating': ['A','T'],
                'calls': {
                    'YRI': (23,3),
                    'CEU': (7,3)
                },
                'outgroup_allele': 'T'
            }}

        The 'calls' entry gives the successful calls in each population, in the
        order that the alleles are specified in 'segregating'.
        Non-diallelic polymorphisms are skipped.

        :param pop_ids: list of which populations to make fs for.
        :param projections: list of sample sizes to project down to for each
            population.
        :param polarized: If True, the data are assumed to be correctly polarized by
            'outgroup_allele'. SNPs in which the 'outgroup_allele'
            information is missing or '-' or not concordant with the
            segregating alleles will be ignored.
            If False, any 'outgroup_allele' info present is ignored,
            and the returned spectrum is folded.
        """
        warnings.warn(
            "Operations using the `data_dict` are deprecated and will be removed "
            "in version 1.5, in favor of `from_vcf` and associated functions "
            "in the `Parsing` module",
            category=DeprecationWarning,
        )

        Npops = len(pop_ids)
        fs = np.zeros(np.asarray(projections) + 1)
        for snp, snp_info in data_dict.items():
            # Skip SNPs that aren't biallelic.
            if len(snp_info["segregating"]) != 2:
                continue

            allele1, allele2 = snp_info["segregating"]
            if not polarized:
                # If we don't want to polarize, we can choose which allele is
                # derived arbitrarily since we'll fold anyways.
                outgroup_allele = allele1
            elif (
                "outgroup_allele" in snp_info
                and snp_info["outgroup_allele"] != "-"
                and snp_info["outgroup_allele"] in snp_info["segregating"]
            ):
                # Otherwise we need to check that it's a useful outgroup
                outgroup_allele = snp_info["outgroup_allele"]
            else:
                # If we're polarized and we didn't have good outgroup info, skip
                # this SNP.
                continue

            # Extract the allele calls for each population.
            allele1_calls = [snp_info["calls"][pop][0] for pop in pop_ids]
            allele2_calls = [snp_info["calls"][pop][1] for pop in pop_ids]
            # How many chromosomes did we call successfully in each population?
            successful_calls = [
                a1 + a2 for (a1, a2) in zip(allele1_calls, allele2_calls)
            ]

            # Which allele is derived (different from outgroup)?
            if allele1 == outgroup_allele:
                derived_calls = allele2_calls
            elif allele2 == outgroup_allele:
                derived_calls = allele1_calls

            # To handle arbitrary numbers of populations in the fs, we need
            # to do some tricky slicing.
            slices = [[np.newaxis] * len(pop_ids) for ii in range(Npops)]
            for ii in range(len(pop_ids)):
                slices[ii][ii] = slice(None, None, None)

            # Do the projection for this SNP.
            pop_contribs = []
            iter = zip(projections, successful_calls, derived_calls)
            for pop_ii, (p_to, p_from, hits) in enumerate(iter):
                contrib = Numerics._cached_projection(p_to, p_from, hits)[
                    tuple(slices[pop_ii])
                ]
                pop_contribs.append(contrib)
            fs += functools.reduce(operator.mul, pop_contribs)
        fsout = Spectrum(fs, mask_corners=mask_corners, pop_ids=pop_ids)
        assert np.all(fsout >= 0)
        if polarized:
            if np.sum(fsout) == 0:
                warnings.warn(
                    "Spectrum is empty. Did you compute the outgroup alleles "
                    "at variable sites?",
                    category=RuntimeWarning,
                )
            return fsout
        else:
            if np.sum(fsout) == 0:
                warnings.warn(
                    "Spectrum is empty. Check input data dictionary.",
                    category=RuntimeWarning,
                )
            return fsout.fold()

    @staticmethod
    def _from_count_dict(count_dict, projections, polarized=True, pop_ids=None):
        """
        Frequency spectrum from data mapping SNP configurations to counts.

        :param count_dict: Result of Misc.count_data_dict
        :param projections: List of sample sizes to project down to for each
            population.
        :param polarized: If True, only include SNPs that count_dict marks as polarized.
            If False, include all SNPs and fold resulting Spectrum.
        :param pop_ids: Optional list of strings containing the population labels.
        """
        warnings.warn(
            "Operations using the `data_dict` are deprecated and will be removed "
            "in version 1.5, in favor of `from_vcf` and associated functions "
            "in the `Parsing` module",
            category=DeprecationWarning,
        )

        # create slices for projection calculation
        slices = [[np.newaxis] * len(projections) for ii in range(len(projections))]
        for ii in range(len(projections)):
            slices[ii][ii] = slice(None, None, None)

        fs_total = moments.Spectrum(
            np.zeros(np.array(projections) + 1), pop_ids=pop_ids
        )
        for (
            (called_by_pop, derived_by_pop, this_snp_polarized),
            count,
        ) in count_dict.items():
            if polarized and not this_snp_polarized:
                continue
            pop_contribs = []
            iter = zip(projections, called_by_pop, derived_by_pop)
            for pop_ii, (p_to, p_from, hits) in enumerate(iter):
                contrib = Numerics._cached_projection(p_to, p_from, hits)[
                    slices[pop_ii]
                ]
                pop_contribs.append(contrib)
            fs_proj = functools.reduce(operator.mul, pop_contribs)

            # create slices for adding projected fs to overall fs
            fs_total += count * fs_proj
        if polarized:
            return fs_total
        else:
            return fs_total.fold()

    @staticmethod
    def _data_by_tri(data_dict):
        """
        Nest the data by derived context and outgroup base.

        The resulting dictionary contains only SNPs which are appropriate for
        use of Hernandez's ancestral misidentification correction. It is
        organized as {(derived_tri, outgroup_base): {snp_id: data,...}}
        """
        warnings.warn(
            "Operations using the `data_dict` are deprecated and will be removed "
            "in version 1.5, in favor of `from_vcf` and associated functions "
            "in the `Parsing` module",
            category=DeprecationWarning,
        )

        result = {}
        genetic_bases = "ACTG"
        for snp, snp_info in data_dict.items():
            # Skip non-diallelic polymorphisms
            if len(snp_info["segregating"]) != 2:
                continue
            allele1, allele2 = snp_info["segregating"]
            # Filter out SNPs where we either non-constant ingroup or outgroup
            # context.
            try:
                ingroup_tri = snp_info["context"]
                outgroup_tri = snp_info["outgroup_context"]
            except KeyError:
                continue
            if not outgroup_tri[1] == snp_info["outgroup_allele"]:
                raise ValueError(
                    "Outgroup context and allele are inconsistent "
                    "for polymorphism: %s." % snp
                )
            outgroup_allele = outgroup_tri[1]

            # These are all the requirements to apply the ancestral correction.
            # First 2 are constant context.
            # Next 2 are sensible context.
            # Next 1 is that outgroup allele is one of the segregating.
            # Next 2 are that segregating alleles are sensible.
            if (
                outgroup_tri[0] != ingroup_tri[0]
                or outgroup_tri[2] != ingroup_tri[2]
                or ingroup_tri[0] not in genetic_bases
                or ingroup_tri[2] not in genetic_bases
                or outgroup_allele not in [allele1, allele2]
                or allele1 not in genetic_bases
                or allele2 not in genetic_bases
            ):
                continue

            if allele1 == outgroup_allele:
                derived_allele = allele2
            elif allele2 == outgroup_allele:
                # In this case, the second allele is non_outgroup
                derived_allele = allele1
            derived_tri = ingroup_tri[0] + derived_allele + ingroup_tri[2]
            result.setdefault((derived_tri, outgroup_allele), {})
            result[derived_tri, outgroup_allele][snp] = snp_info
        return result

    @staticmethod
    def _from_data_dict_corrected(
        data_dict, pop_ids, projections, fux_filename, force_pos=True, mask_corners=True
    ):
        """
        Spectrum from a dictionary of polymorphisms, corrected for ancestral
        misidentification.

        The correction is based upon Hernandez, Williamson & Bustamante *Mol Biol Evol*
        24:1792 (2007)

        :param force_pos: If the correction is too agressive, it may leave some small
            entries in the fs less than zero. If force_pos is true,
            these entries will be set to zero, in such a way that the
            total number of segregating SNPs is conserved.
        :param fux_filename: The name of the file containing the
            misidentification probabilities.
            The file is of the form:
            - # Any number of comments lines beginning with #
            - AAA T 0.001
            - AAA G 0.02
            - ...
            Where every combination of three + one bases is considered
            (order is not important).  The triplet is the context and
            putatively derived allele (x) in the reference species. The
            single base is the base (u) in the outgroup. The numerical
            value is 1-f_{ux} in the notation of the paper.

        The data dictionary should be organized as:
            {snp_id:{'segregating': ['A','T'],
                     'calls': {'YRI': (23,3),
                                'CEU': (7,3)
                                },
                     'outgroup_allele': 'T',
                     'context': 'CAT',
                     'outgroup_context': 'CAT'
                    }
            }
        The additional entries are 'context', which includes the two flanking
        bases in the species of interest, and 'outgroup_context', which
        includes the aligned bases in the outgroup.

        This method skips entries for which the correction cannot be applied.
        Most commonly this is because of missing or non-constant context.
        """
        warnings.warn(
            "Operations using the `data_dict` are deprecated and will be removed "
            "in version 1.5, in favor of `from_vcf` and associated functions "
            "in the `Parsing` module",
            category=DeprecationWarning,
        )

        # Read the fux file into a dictionary.
        fux_dict = {}
        f = open(fux_filename)
        for line in f.readlines():
            if line.startswith("#"):
                continue
            sp = line.split()
            fux_dict[(sp[0], sp[1])] = 1 - float(sp[2])
        f.close()

        # Divide the data into classes based on ('context', 'outgroup_allele')
        by_context = Spectrum._data_by_tri(data_dict)

        fs = np.zeros(np.asarray(projections) + 1)
        while by_context:
            # Each time through this loop, we eliminate two entries from the
            # data dictionary. These correspond to one class and its
            # corresponding misidentified class.
            (derived_tri, out_base), nomis_data = by_context.popitem()

            # The corresponding bases if the ancestral state had been
            # misidentifed.
            mis_out_base = derived_tri[1]
            mis_derived_tri = derived_tri[0] + out_base + derived_tri[2]
            # Get the data for that case. Note that we default to an empty
            # dictionary if we don't have data for that class.
            mis_data = by_context.pop((mis_derived_tri, mis_out_base), {})

            fux = fux_dict[(derived_tri, out_base)]
            fxu = fux_dict[(mis_derived_tri, mis_out_base)]

            # Get the spectra for these two cases
            Nux = Spectrum.from_data_dict(nomis_data, pop_ids, projections)
            Nxu = Spectrum.from_data_dict(mis_data, pop_ids, projections)

            # Equations 5 & 6 from the paper.
            Nxu_rev = Numerics.reverse_array(Nxu)
            Rux = (fxu * Nux - (1 - fxu) * Nxu_rev) / (fux + fxu - 1)
            Rxu = Numerics.reverse_array(
                (fux * Nxu_rev - (1 - fux) * Nux) / (fux + fxu - 1)
            )

            fs += Rux + Rxu

        # Here we take the negative entries, and flip them back, so they end up
        # zero and the total number of SNPs is conserved.
        if force_pos:
            negative_entries = np.minimum(0, fs)
            fs -= negative_entries
            fs += Numerics.reverse_array(negative_entries)

        return Spectrum(fs, mask_corners=mask_corners, pop_ids=pop_ids)

    @staticmethod
    def from_vcf(
        vcf_file,
        pop_mapping=None,
        pop_file=None,
        pops=None,
        bed_file=None,
        interval=None,
        anc_seq_file=None,
        allow_low_confidence=False,
        use_AA=False,
        filters=None,
        allow_multiallelic=False,
        sample_sizes=None,
        mask_corners=True,
        ploidy=2,
        verbose=0,
        folded=False,
    ):
        """
        Compute the SFS from genotype data stored in a VCF file. There are
        several optional parameters for controlling the assignment of ancestral
        states, filtering by quality or annotation, restricting SFS parsing to
        certain intervals, and adjusting the shape of the output SFS.

        There are several ways to specify estimated ancestral states. By
        default, VCF ``REF`` alleles are interpreted as ancestral alleles. In
        this case the output SFS should be folded, because the reference allele
        does not in general correspond to the ancestral allele. If ``use_AA``
        is True, then the ``INFO/AA`` VCF field specifies the ancestral allele.
        Sites where ``INFO/AA`` is absent or missing data (represented by '.')
        are skipped, raising a warning at the first occurence.

        If ``anc_seq_file`` (FASTA format) is given, then ancestral alleles are
        read from it. Here also, sites are skipped when they lack a valid
        ancestral allele. This behavior is modulated by
        ``allow_low_confidence``: when True, sites assigned ancestral states
        represented in lower-case (e.g. 'a') are retained. This usually denotes
        a low-confidence assignment. Otherwise such sites are skipped. If there
        are sites in the VCF that fall beyond the end of the FASTA sequence
        which are not otherwise masked or excluded by the ``interval`` argument,
        an error is raised.

        When ``allow_multiallelic`` is False and the ancestral allele is not
        represented as either the reference or alternate allele at a site, that
        site is skipped automatically. The relationship between derived alleles
        at such sites is not generally clear.

        The parameter ``filters`` allows filtering by quality and/or annotation
        at the site and sample level. It should be a flat dictionary with str
        keys. Key-value pairs may have the following forms:
        
        - 'QUAL' should map to a single number (float or int), imposing a minimum
          value for site ``QUAL`` fields.
        - 'FILTER' should map to a string or a set, tuple or list of strings. For
          a site to pass, its ``FILTER`` field must equal either the value of
          'FILTER' (if a string) or that of one of its elements (if a set,
          tuple or list).
        - 'INFO/FIELD' e.g. 'INFO/GQ' may map to a number, a string, or a set,
          tuple or list of strings. When it maps to a number, passing sites
          must have ``INFO/FIELD`` greater than or equal to that number to
          pass. When it maps to a string, sites must have equal
          ``INFO/FIELD``. When it maps to a set, tuple or list of strings, the
          filter is said to be categorical and a site's ``INFO/FIELD`` must be
          a member of the set/tuple/list to pass.
          The types of values are not explictly checked against the proper
          type for their fields- e.g. if 'INFO/GQ' maps to a string rather
          than a number, no explicit warning is raised, although an error will
          typically be thrown once parsing begins.
        - 'SAMPLE/FIELD' e.g. 'SAMPLE/GQ' imposes filters at the sample level.
          'FORMAT/FIELD' is equivalent to 'SAMPLE/FIELD'. The 'FIELD' should
          correspond to an entry in the ``FORMAT`` column of the VCF file.
          Typing is the same as for ``INFO/FIELD``.

        When fields targeted for filtering are missing ('.') or absent in given
        lines/samples, those lines/samples are not skipped, but a one-time
        alert message is raised. Depending on the context, this may be a sign
        of misspecified filters, or it may be unproblematic. Any combination of
        valid filter fields is permissible.

        :param vcf_file: Pathname of the VCF file to parse. The file may be
            gzipped, bgzipped or uncompressed.
        :type vcf_file: str
        :param pop_mapping: Optional dictionary (default None) mapping
            population IDs to lists of VCF sample IDs. Equivalent in function
            to, and mutually exclusive with, ``pop_file``.
        :type pop_mapping: dict, optional
        :param pop_file: Pathname of a whitespace-separated file mapping samples
            to populations with the format SAMPLE POPULATION. Sample names must
            be unique and there should be one of them on each line (default None
            combines all samples into a single population 'ALL'). Samples
            present in the VCF but not included here are ignored.
        :type pop_file: str, optional
        :param pops: A list of populations from ``pop_file`` to parse (default
            None). Only functions when ``pop_file`` is given. Populations not in
            ``pops`` are ignored. If None, then all populations in ``pop_file`` are
            included.
        :type pops: list of str
        :param bed_file: Pathname of a BED file defining the intervals within
            which to parse; useful for applying masks to exclude difficult-to-
            call or functionally constrained genomic regions (default None).
            BED files represent intervals as 0-indexed and half-open (the ends
            of intervals are noninclusive).
        :type bed_file: str, optional
        :param interval: 2-tuple or 2-list specifying a 1-indexed, half-open
            (upper boundary noninclusive) genomic window to parse (default
            None). May be used in conjuction with a BED file.
        :type interval: tuple or list of integers, optional
        :param anc_seq_file: Pathname of a FASTA file defining inferred
            ancestral nucleotide states (default None).
        :type anc_seq_file: str, optional
        :param allow_low_confidence: If True (default False) and ``anc_seq_file``
            is given, allows low-confidence ancestral state assignments-
            represented by lower-case nucleotide codes- to stand. If False,
            sites with low-confidence assignments are skipped.
        :type allow_low_confidence: bool, optional
        :param use_AA: If True, use entries in the VCF field ``INFO/AA`` to
            assign ancestral alleles (default False).
        :type use_AA: bool, optional
        :param filters: A dictionary mapping VCF fields to filter criteria, for
            imposing quantitative thresholds on measures of genotype quality and
            categorical requirements on annotations. Filtering is discussed
            above.
        :type filters: dict, optional
        :param allow_multiallelic: If True (default False), includes sites with
            more than one alternate allele, counting each derived allele at such
            sites as a separate entry in the SFS- otherwise multiallelic sites
            are skipped. Also allows sites where neither the reference nor any
            alternate allelle(s) matches the assigned ancestral state, which are
            skipped when False.
        :type allow_multiallelic: bool, optional
        :param sample_sizes: Dictionary mapping populations to haploid sample
            sizes (default None). Determines the shape of the returned SFS.
            Any VCF sites with sample sizes greater than ``sample_sizes`` will be
            projected down to match it. This may be useful when some genotype
            data is missing or filtered- sites with missing data are otherwise
            not included in the output SFS. When not given, output sample sizes
            default to the sample sizes implied by ``ploidy`` and the number of
            individuals in each population.
        :type sample_sizes: dict, optional
        :param mask_corners: If True (default), the 'observed in none' and
            'observed in all' entries of the SFS array are masked.
        :type mask_corners: bool, optional
        :param ploidy: Optionally defines the ploidy of samples (default 2).
            Used to determine the haploid sample size from the number of sampled
            individuals when ``sample_sizes`` is not given.
        :type plody: int, optional
        :param verbose: If > 0, print a progress message every ``verbose`` lines
            (default 0).
        :type verbose: int, optional
        :param folded: If True, return the folded SFS (default False).
        :type folded: bool, optional
        :returns: The SFS, represented as a ``moments.Spectrum`` instance.
        :rtype: moments.Spectrum
        """
        fs = Parsing.parse_vcf(
            vcf_file,
            pop_mapping=pop_mapping,
            pop_file=pop_file,
            pops=pops,
            bed_file=bed_file,
            interval=interval,
            anc_seq_file=anc_seq_file,
            allow_low_confidence=allow_low_confidence,
            use_AA=use_AA,
            filters=filters,
            allow_multiallelic=allow_multiallelic,
            ploidy=ploidy,
            verbose=verbose,
            sample_sizes=sample_sizes,
            mask_corners=mask_corners,
            folded=folded,
        )
        return fs

    @staticmethod
    def from_demes(
        g,
        sampled_demes=None,
        sample_sizes=None,
        sample_times=None,
        samples=None,
        unsampled_n=4,
        gamma=None,
        h=None,
        theta=1,
    ):
        """
        Takes a deme graph and computes the SFS. ``demes`` is a package for
        specifying demographic models in a user-friendly, human-readable YAML
        format. This function automatically parses the demographic description
        and returns a SFS for the specified populations and sample sizes.

        .. note::

            If a deme sample time is requested that is earlier than the deme's
            end time, for example to simulate ancient samples, we must create a
            new population for that ancient sample. This can cause large
            slow-downs, as the computation cost of computing the SFS grows
            quickly in the number of populations.

        :param g: A ``demes`` DemeGraph from which to compute the SFS. The DemeGraph
            can either be specified as a YAML file, in which case ``g`` is a string,
            or as a ``DemeGraph`` object.
        :type g: str or :class:`demes.DemeGraph`
        :param sampled_demes: A list of deme IDs to take samples from. We can repeat
            demes, as long as the sampling of repeated deme IDs occurs at distinct
            times.
        :type sampled_demes: list of strings
        :param sample_sizes: A list of the same length as ``sampled_demes``,
            giving the sample sizes for each sampled deme.
        :type sample_sizes: list of ints
        :param sample_times: If None, assumes all sampling occurs at the end of the
            existence of the sampled deme. If there are
            ancient samples, ``sample_times`` must be a list of same length as
            ``sampled_demes``, giving the sampling times for each sampled
            deme. Sampling times are given in time units of the original deme graph,
            so might not necessarily be generations (e.g. if ``g.time_units`` is years)
        :type sample_times: list of floats, optional
        :param unsampled_n: The default sample size of unsampled demes, which must be
            greater than or equal to 4.
        :type unsampled_n: int, optional
        :param gamma: The scaled selection coefficient(s), 2*Ne*s. Defaults to None,
            which implies neutrality. Can be given as a scalar value, in which case
            all populations have the same selection coefficient. Alternatively, can
            be given as a dictionary, with keys given as population names in the
            input Demes model. Any population missing from this dictionary will be
            assigned a selection coefficient of zero. A non-zero default selection
            coefficient can be provided, using the key ``_default``. See the Demes
            exension documentation for more details and examples.
        :type gamma: float or dict
        :param h: The dominance coefficient(s). Defaults to additivity (or genic
            selection). Can be given as a scalar value, in which case all populations
            have the same dominance coefficient. Alternatively, can be given as a
            dictionary, with keys given as population names in the input Demes model.
            Any population missing from this dictionary will be assigned a dominance
            coefficient of 1/2 (additivity). A different default dominance
            coefficient can be provided, using the key ``_default``. See the Demes
            exension documentation for more details and examples.
        :type h: float or dict
        :param theta: The population-size scaled mutation rate (4*Ne*u or 4*Ne*u*L).
            The default value is 1. For more control of mutation rates and mutation
            models (including using a reversible mutation model), please use
            ``moments.Demes.SFS``, which has options to specify theta, the per-base
            mutation rate u, and/or a reversible mutation model that allows for
            different forward and backward mutation rates.
        :type theta: scalar
        :return: A ``moments`` site frequency spectrum, with dimension equal to the
            length of ``sampled_demes``, and shape equal to ``sample_sizes`` plus one
            in each dimension, indexing the allele frequency in each deme from 0
            to n[i], where i is the deme index.
        :rtype: :class:`moments.Spectrum`
        """
        if isinstance(g, str):
            dg = demes.load(g)
        else:
            dg = g

        fs = moments.Demes.Demes.SFS(
            dg,
            sampled_demes,
            sample_sizes,
            sample_times=sample_times,
            samples=samples,
            unsampled_n=unsampled_n,
            gamma=gamma,
            h=h,
            theta=theta,
        )
        return fs

    @staticmethod
    def from_angsd(
        sfs_file, sample_sizes, pop_ids=None, folded=False, mask_corners=True
    ):
        """
        Convert ANGSD output to a moments Spectrum object. The sample sizes
        are given as number of haploid genome copies (twice the number of
        sampled diploid individuals).

        :param string sfs_file: The n-dimensional SFS from ANGSD. This should be a
            file with a single line of numbers, as entries in the SFS.
        :param list sample_sizes: A list of integers with length equal to the number
            of population, storing the haploid sample size in each population.
            The order must match the population order provided to ANGSD.
        :param list pop_ids: A list of strings equal with length equal to the number
            of population, specifying the population name for each.
        :param bool folded: If False (default), we assume ancestral states are
            known, returning an unfolded SFS. If True, the returned SFS is folded.
        :param bool mask_corners: If True (default), mask the fixed bins in the
            SFS. If False, the fixed bins will remain unmasked.
        :return: A ``moments`` site frequency spectrum.
        :rtype: :class:`moments.Spectrum`
        """
        if pop_ids is not None and len(pop_ids) != len(sample_sizes):
            raise ValueError(
                "length of pop ids must equal the number of "
                "populations given by sample sizes"
            )

        if len(sample_sizes) > 2:
            warnings.warn(
                "Importing ANGSD-formatted data has only been tested for 2 populations",
                category=UserWarning,
            )
        # get the SFS data
        with open(sfs_file) as f_data:
            data_line = f_data.readline()

        data = np.array([float(d) for d in data_line.split()])

        # check that the length of data matches the number of bins in the SFS
        reshape_dims = [n + 1 for n in sample_sizes]
        if np.prod(reshape_dims) != len(data):
            raise ValueError("size of data does not match input sample sizes")

        data = np.reshape(data, reshape_dims)

        # turn into moments object
        fs = moments.Spectrum(data, mask_corners=mask_corners, pop_ids=pop_ids)
        if folded:
            fs = fs.fold()
        return fs

    # The code below ensures that when I do arithmetic with Spectrum objects,
    # it is not done between a folded and an unfolded array. If it is, I raise
    # a ValueError.

    # While I'm at it, I'm also fixing the annoying behavior that if a1 and a2
    # are masked arrays, and a3 = a1 + a2. Then wherever a1 or a2 was masked,
    # a3.data ends up with the a1.data values, rather than a1.data + a2.data.
    # Note that this fix doesn't work for operation by np.ma.exp and
    # np.ma.log. Guess I can't have everything.

    # I'm using exec here to avoid copy-pasting a dozen boiler-plate functions.
    # The calls to check_folding_equal ensure that we don't try to combine
    # folded and unfolded Spectrum objects.

    # I set check_folding = False in the constructor because it raises useless
    # warnings when, for example, I do (model + 1).

    # These functions also ensure that the pop_ids
    # get properly copied over.

    # This is pretty advanced Python voodoo, so don't fret if you don't
    # understand it at first glance. :-)
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
    newpop_ids = self.pop_ids
    if hasattr(other, 'pop_ids'):
        if other.pop_ids is None:
            newpop_ids = self.pop_ids
        elif self.pop_ids is None:
            newpop_ids = other.pop_ids
        elif other.pop_ids != self.pop_ids:
            logger.warn('Arithmetic between Spectra with different pop_ids. '
                        'Resulting pop_id may not be correct.')
    outfs = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   mask_corners=False, data_folded=self.folded,
                                   check_folding=False, pop_ids=newpop_ids)
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
    if hasattr(other, 'pop_ids') and other.pop_ids is not None\
             and other.pop_ids != self.pop_ids:
        logger.warn('Arithmetic between Spectra with different pop_ids. '
                    'Resulting pop_id may not be correct.')
    return self
"""
            % {"method": method}
        )

    def _check_other_folding(self, other):
        """
        Ensure other Spectrum has same .folded status
        """
        if isinstance(other, self.__class__) and other.folded != self.folded:
            raise ValueError(
                "Cannot operate with a folded Spectrum and an " "unfolded one."
            )


# Allow spectrum objects to be pickled.
# See http://effbot.org/librarybook/copy-reg.htm
try:
    import copy_reg

    def Spectrum_unpickler(data, mask, data_folded, pop_ids):
        return moments.Spectrum(
            data,
            mask,
            mask_corners=False,
            data_folded=data_folded,
            check_folding=False,
            pop_ids=pop_ids,
        )

    def Spectrum_pickler(fs):
        return (
            Spectrum_unpickler,
            (fs.data, fs.mask, fs.folded, fs.pop_ids),
        )

    copy_reg.pickle(Spectrum, Spectrum_pickler, Spectrum_unpickler)
except:
    import copyreg

    def Spectrum_unpickler(data, mask, data_folded, pop_ids):
        return moments.Spectrum(
            data,
            mask,
            mask_corners=False,
            data_folded=data_folded,
            check_folding=False,
            pop_ids=pop_ids,
        )

    def Spectrum_pickler(fs):
        return (
            Spectrum_unpickler,
            (fs.data, fs.mask, fs.folded, fs.pop_ids),
        )

    copyreg.pickle(Spectrum, Spectrum_pickler, Spectrum_unpickler)
