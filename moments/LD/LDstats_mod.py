import logging

logging.basicConfig()
logger = logging.getLogger("LDstats_mod")

import os, sys
import numpy, numpy as np
import copy
import demes
import warnings

from . import Numerics, Util
import moments.Demes.Demes


class LDstats(list):
    """
    Represents linkage disequilibrium statistics as a list of arrays, where
    each entry in the list is an array of statistics for a corresponding
    recombination rate. The final entry in the list is always the heterozygosity
    statistics. Thus, if we have an LDstats object for 3 recombination rate
    values, the list will have length 4.

    LDstats are represented as a list of statistics over two locus pairs for
    a given recombination distance.

    :param data: A list of LD and heterozygosity stats.
    :type data: list of arrays
    :param num_pops: Number of populations. For one population, higher order
        statistics may be computed.
    :type num_pops: int
    :param pop_ids: Population IDs in order that statistics are represented here.
    :type pop_ids: list of strings, optional
    """

    def __new__(self, data, num_pops=None, pop_ids=None):
        if num_pops == None:
            raise ValueError("Specify number of populations as num_pops=.")
        my_list = super(LDstats, self).__new__(self, data, num_pops=None, pop_ids=None)

        if hasattr(data, "num_pops"):
            my_list.num_pops = data.num_pops
        else:
            my_list.num_pops = num_pops

        if hasattr(data, "pop_ids"):
            my_list.pop_ids = data.pop_ids
        else:
            my_list.pop_ids = pop_ids

        return my_list

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], "__iter__"):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    # See http://www.scipy.org/Subclasses for information on the
    # __array_finalize__ and __array_wrap__ methods. I had to do some debugging
    # myself to discover that I also needed _update_from.
    # Also, see http://docs.scipy.org/doc/numpy/reference/arrays.classes.html
    # Also, see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    #
    # We need these methods to ensure extra attributes get copied along when
    # we do arithmetic on the LD stats.
    def __array_finalize__(self, obj):
        if obj is None:
            return
        np.ma.masked_array.__array_finalize__(self, obj)
        self.num_pops = getattr(obj, "num_pops", "unspecified")
        self.pop_ids = getattr(obj, "pop_ids", "unspecified")

    def __array_wrap__(self, obj, context=None):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, context=context)
        result.num_pops = self.num_pops
        result.pop_ids = self.pop_ids
        return result

    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, "num_pops"):
            self.num_pops = obj.num_pops
        if hasattr(obj, "pop_ids"):
            self.pop_ids = obj.pop_ids

    # masked_array has priority 15.
    __array_priority__ = 20

    def __repr__(self):
        ys = np.asanyarray(self[:-1])
        h = self[-1]
        return "LDstats(%s, %s, num_pops=%s, pop_ids=%s)" % (
            str(ys),
            str(h),
            str(self.num_pops),
            str(self.pop_ids),
        )

    def names(self):
        """
        Returns the set of LD and heterozygosity statistics names for the
        number of populations represented by the LDstats.

        Note that this will always return the full set of statistics,
        """
        if self.num_pops == None:
            raise ValueError(
                "Number of populations must be specified (as stats.num_pops)"
            )

        return Util.moment_names(
            self.num_pops
        )  # returns (ld_stat_names, het_stat_names)

    def LD(self, pops=None):
        """
        Returns LD stats for populations given (if None, returns all).

        :param pops: The indexes of populations to return stats for.
        :type pops: list of ints, optional
        """
        if len(self) <= 1:
            raise ValueError("No LD statistics present")
        else:
            if pops is not None:
                to_marginalize = list(set(range(self.num_pops)) - set(pops))
                Y_new = self.marginalize(to_marginalize)
                return np.array(Y_new[:-1])
            else:
                return np.array(self[:-1])

    def H(self, pops=None):
        """
        Returns heterozygosity statistics for the populations given.

        :param pops: The indexes of populations to return stats for.
        :type pops: list of ints, optional
        """
        if pops is not None:
            to_marginalize = list(set(range(self.num_pops)) - set(pops))
            Y_new = self.marginalize(to_marginalize)
            return Y_new[-1]
        else:
            return self[-1]

    def f2(self, X, Y):
        """
        Returns :math:`f_2(X, Y) = (X-Y)^2`.

        X, and Y can be specified as population ID strings, or as indexes
        (but these cannot be mixed).

        :param X: One of the populations, as index or population ID.
        :param Y: The other population, as index or population ID.
        """
        if type(X) is str:
            if type(Y) is not str:
                raise ValueError("X and Y must both be strings or both be indexes")
            if X not in self.pop_ids:
                raise ValueError(f"Population {X} not in pop_ids: {self.pop_ids}")
            if Y not in self.pop_ids:
                raise ValueError(f"Population {X} not in pop_ids: {self.pop_ids}")
            X = self.pop_ids.index(X)
            Y = self.pop_ids.index(Y)
        elif type(X) is int:
            if type(Y) is not int:
                raise ValueError("X and Y must both be strings or both be indexes")
            if X < 0 or Y < 0 or X >= self.num_pops or Y >= self.num_pops:
                raise ValueError("Population indexes out of bounds")
        else:
            raise ValueError("X and Y must both be strings or both be indexes")

        stats = self.names()[1]
        if X > Y:
            X, Y = Y, X
        H_X = self.H()[stats.index(f"H_{X}_{X}")]
        H_Y = self.H()[stats.index(f"H_{Y}_{Y}")]
        H_XY = self.H()[stats.index(f"H_{X}_{Y}")]
        return H_XY - H_X / 2 - H_Y / 2

    def f3(self, X, Y, Z):
        """
        Returns :math:`f_3(X; Y, Z) = (X-Y)(X-Z)`. A significantly negative
        :math:`f_3` of this form suggests that population X is the result
        of admixture between ancient populations related to Y and Z. A positive
        value suggests that X is an outgroup to Y and Z.

        X, Y, and Z can be specified as population ID strings, or as indexes
        (but these cannot be mixed).

        :param X: The "test" population, as index or population ID.
        :param Y: The first reference population, as index or population ID.
        :param Z: The second reference population, as index or population ID.
        """
        if type(X) is str:
            if type(Y) is not str or type(Z) is not str:
                raise ValueError("Inputs must be all strings or all indexes")
            for _ in [X, Y, Z]:
                if _ not in self.pop_ids:
                    raise ValueError(f"Population {_} not in pop_ids: {self.pop_ids}")
            X = self.pop_ids.index(X)
            Y = self.pop_ids.index(Y)
            Z = self.pop_ids.index(Z)
        elif type(X) is int:
            if type(Y) is not int or type(Z) is not int:
                raise ValueError("Inputs must be all strings or all indexes")
            if (
                X < 0
                or Y < 0
                or Z < 0
                or X >= self.num_pops
                or Y >= self.num_pops
                or Z >= self.num_pops
            ):
                raise ValueError("Population indexes out of bounds")
        else:
            raise ValueError("Inputs must be all strings or all indexes")

        stats = self.names()[1]
        H_XX = self.H()[stats.index(Util.map_moment(f"H_{X}_{X}"))]
        H_XY = self.H()[stats.index(Util.map_moment(f"H_{X}_{Y}"))]
        H_XZ = self.H()[stats.index(Util.map_moment(f"H_{X}_{Z}"))]
        H_YZ = self.H()[stats.index(Util.map_moment(f"H_{Y}_{Z}"))]

        return (H_XY + H_XZ - H_XX - H_YZ) / 2

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
        if type(X) is str:
            if type(Y) is not str or type(Z) is not str or type(W) is not str:
                raise ValueError("Inputs must be all strings or all indexes")
            for _ in [X, Y, Z, W]:
                if _ not in self.pop_ids:
                    raise ValueError(f"Population {_} not in pop_ids: {self.pop_ids}")
            X = self.pop_ids.index(X)
            Y = self.pop_ids.index(Y)
            Z = self.pop_ids.index(Z)
            W = self.pop_ids.index(W)
        elif type(X) is int:
            if type(Y) is not int or type(Z) is not int or type(W) is not int:
                raise ValueError("Inputs must be all strings or all indexes")
            if (
                X < 0
                or Y < 0
                or Z < 0
                or W < 0
                or X >= self.num_pops
                or Y >= self.num_pops
                or Z >= self.num_pops
                or W >= self.num_pops
            ):
                raise ValueError("Population indexes out of bounds")
        else:
            raise ValueError("Inputs must be all strings or all indexes")

        stats = self.names()[1]
        H_XW = self.H()[stats.index(Util.map_moment(f"H_{X}_{W}"))]
        H_XZ = self.H()[stats.index(Util.map_moment(f"H_{X}_{Z}"))]
        H_YW = self.H()[stats.index(Util.map_moment(f"H_{Y}_{W}"))]
        H_YZ = self.H()[stats.index(Util.map_moment(f"H_{Y}_{Z}"))]

        return (H_XW - H_XZ - H_YW + H_YZ) / 2

    def H2(self, X, Y=None, phased=True):
        """
        Note: the H2 name may change! This is sometimes called "D+".

        This is the statistics E[2*fAB*fab + 2*fAb*faB], which measures
        the probability of polymorphism between two sampled genome copies
        at two loci.

        This is closely related to pi2=p(1-p)q(1-q), which is inherently
        a four-haplotype statistic. Instead, H2 is a two-haplotype
        statistic, which can be measured with just a single diploid.

        In the one population case, it equals 4D^2+2Dz+4pi2. In the two
        population case, it equals 4D1*D2+Dz(1,2,2)+Dz(2,1,1)+4pi2(1,2,1,2).

        At steady state, the solution is
        theta**2 * (36 + 14*rho + rho**2) / (18 + 13*rho + rho**2).

        To compare to unphased data estimated assuming that double
        heterozygotes have equal probability of being in coupling or
        repulsion LD, the statistic equals
        D1*D2+1/2(Dz(1,2,2)+Dz(2,1,1)+4pi2.

        Note: This unphased case (setting phased=False) is only relevant to
        comparing to data where a single diploid individual exists from each
        population, and it is only needed for cross-population H2!
        """
        if type(X) is str:
            if X not in self.pop_ids:
                raise ValueError(f"Population {X} not in pop_ids")
            X = self.pop_ids.index(X)

        if Y is None:
            Y = X
        else:
            if type(Y) is str:
                if Y not in self.pop_ids:
                    raise ValueError(f"Population {Y} not in pop_ids")
                Y = self.pop_ids.index(Y)

        if X < 0 or X >= self.num_pops or Y < 0 or Y >= self.num_pops:
            raise ValueError("Population indexes out of bounds")

        if Y < X:
            X, Y = Y, X

        DD = self.names()[0].index(f"DD_{X}_{Y}")
        Dz0 = self.names()[0].index(f"Dz_{X}_{Y}_{Y}")
        Dz1 = self.names()[0].index(f"Dz_{Y}_{X}_{X}")
        pi2 = self.names()[0].index(f"pi2_{X}_{Y}_{X}_{Y}")
        data = self.LD()
        if phased:
            Dplus = 4 * data[:, DD] + data[:, Dz0] + data[:, Dz1] + 4 * data[:, pi2]
        else:
            Dplus = (
                data[:, DD]
                + 1 / 2 * data[:, Dz0]
                + 1 / 2 * data[:, Dz1]
                + 4 * data[:, pi2]
            )
        return Dplus

    # demographic and manipulation functions
    def split(self, pop_to_split, new_ids=None):
        """
        Splits the population given into two child populations. One child
        population keeps the same index and the second child population is
        placed at the end of the list of present populations. If new_ids
        is given, we can set the population IDs of the child populations,
        but only if the input LDstats have population IDs available.

        :param pop_to_split: The index of the population to split.
        :type pop_to_split: int
        :param new_ids: List of child population names, of length two.
        :type new_ids: list of strings, optional
        """
        if type(pop_to_split) is not int:
            raise ValueError("population to split must be an integer index")
        if pop_to_split < 0:
            raise ValueError("population to split must be a nonnegative index")
        if pop_to_split >= self.num_pops:
            raise ValueError("population to split is greater than maximum index")

        h = self[-1]
        ys = self[:-1]

        h_new = Numerics.split_h(h, pop_to_split, self.num_pops)
        ys_new = []
        for y in ys:
            ys_new.append(Numerics.split_ld(y, pop_to_split, self.num_pops))

        if self.pop_ids is not None and new_ids is not None:
            new_pop_ids = copy.copy(self.pop_ids)
            new_pop_ids[pop_to_split] = new_ids[0]
            new_pop_ids.append(new_ids[1])
        else:
            new_pop_ids = None

        return LDstats(
            ys_new + [h_new], num_pops=self.num_pops + 1, pop_ids=new_pop_ids
        )

    def swap_pops(self, pop0, pop1):
        """
        Swaps pop0 and pop1 in the order of the population in the LDstats.

        :param pop0: The index of the first population to swap.
        :type pop0: int
        :param pop1: The index of the second population to swap.
        :type pop1: int
        """
        if pop0 >= self.num_pops or pop1 >= self.num_pops or pop0 < 0 or pop1 < 0:
            raise ValueError("Invalid population number specified.")
        if pop0 == pop1:
            return self

        mom_list = Util.moment_names(self.num_pops)
        h_new = np.zeros(len(mom_list[-1]))

        if len(self) == 2:
            y_new = np.zeros(len(mom_list[0]))
        if len(self) > 2:
            ys_new = [np.zeros(len(mom_list[0])) for i in range(len(self) - 1)]

        pops_old = list(range(self.num_pops))
        pops_new = list(range(self.num_pops))
        pops_new[pop0] = pop1
        pops_new[pop1] = pop0

        d = dict(zip(pops_old, pops_new))

        # swap heterozygosity statistics
        for ii, mom in enumerate(mom_list[-1]):
            pops_mom = [int(p) for p in mom.split("_")[1:]]
            pops_mom_new = [d.get(p) for p in pops_mom]
            mom_new = mom.split("_")[0] + "_" + "_".join([str(p) for p in pops_mom_new])
            h_new[ii] = self[-1][mom_list[-1].index(Util.map_moment(mom_new))]

        # swap LD statistics
        if len(self) > 1:
            for ii, mom in enumerate(mom_list[0]):
                pops_mom = [int(p) for p in mom.split("_")[1:]]
                pops_mom_new = [d.get(p) for p in pops_mom]
                mom_new = (
                    mom.split("_")[0] + "_" + "_".join([str(p) for p in pops_mom_new])
                )
                if len(self) == 2:
                    y_new[ii] = self[0][mom_list[0].index(Util.map_moment(mom_new))]
                elif len(self) > 2:
                    for jj in range(len(self) - 1):
                        ys_new[jj][ii] = self[jj][
                            mom_list[0].index(Util.map_moment(mom_new))
                        ]

        if self.pop_ids is not None:
            current_order = self.pop_ids
            new_order = [self.pop_ids[d[ii]] for ii in range(self.num_pops)]
        else:
            new_order = None

        if len(self) == 1:
            return LDstats([h_new], num_pops=self.num_pops, pop_ids=new_order)
        elif len(self) == 2:
            return LDstats([y_new, h_new], num_pops=self.num_pops, pop_ids=new_order)
        else:
            return LDstats(ys_new + [h_new], num_pops=self.num_pops, pop_ids=new_order)

    def marginalize(self, pops):
        """
        Marginalize over the LDstats, removing moments for given populations.

        :param pops: The index or list of indexes of populations to marginalize.
        :type pops: int or list of ints
        """
        if hasattr(pops, "__len__") == False:
            pops = [pops]
        if self.num_pops == len(pops):
            raise ValueError("Marginalization would remove all populations.")

        # multiple pop indices to marginalize over
        names_from_ld, names_from_h = Util.moment_names(self.num_pops)
        names_to_ld, names_to_h = Util.moment_names(self.num_pops - len(pops))
        y_new = [np.zeros(len(names_to_ld)) for i in range(len(self) - 1)] + [
            np.zeros(len(names_to_h))
        ]
        count = 0
        for mom in names_from_ld:
            mom_pops = [int(p) for p in mom.split("_")[1:]]
            if len(np.intersect1d(pops, mom_pops)) == 0:
                for ii in range(len(y_new) - 1):
                    y_new[ii][count] = self[ii][names_from_ld.index(mom)]
                count += 1
        count = 0
        for mom in names_from_h:
            mom_pops = [int(p) for p in mom.split("_")[1:]]
            if len(np.intersect1d(pops, mom_pops)) == 0:
                y_new[-1][count] = self[-1][names_from_h.index(mom)]
                count += 1

        if self.pop_ids == None:
            return LDstats(y_new, num_pops=self.num_pops - len(pops))
        else:
            new_ids = copy.copy(self.pop_ids)
            for ii in sorted(pops)[::-1]:
                new_ids.pop(ii)
            return LDstats(y_new, num_pops=self.num_pops - len(pops), pop_ids=new_ids)

    ## Admix takes two populations, creates new population with fractions f, 1-f
    ## Merge takes two populations (pop1, pop2) and merges together with fraction
    ## f from population pop1, and (1-f) from pop2
    ## Pulse migrate again takes two populations, pop1 and pop2, and replaces fraction
    ## f from pop2 with pop1
    ## In each case, we use the admix function, which appends a new population on the end
    ## In the case of merge, we use admix and then marginalize pop1 and pop2
    ## in the case of pulse migrate, we use admix, and then swap new pop with pop2, and
    ## then marginalize pop2, so the new population takes the same position in pop_ids
    ## that pop2 was previously in

    def admix(self, pop0, pop1, f, new_id="Adm"):
        """
        Admixture between pop0 and pop1, given by indexes. f is the fraction
        contributed by pop0, so pop1 contributes 1-f. If new_id is not specified,
        the admixed population's name is 'Adm'. Otherwise, we can set it with
        new_id=new_pop_id.

        :param pop0: First population to admix.
        :type pop0: int
        :param pop1: Second population to admix.
        :type pop1: int
        :param f: The fraction of ancestry contributed by pop0, so pop1 contributes
            1 - f.
        :type f: float
        :param new_id: The name of the admixed population.
        :type new_id: str, optional
        """
        if (
            self.num_pops < 2
            or pop0 >= self.num_pops
            or pop1 >= self.num_pops
            or pop0 < 0
            or pop1 < 0
        ):
            raise ValueError("Improper usage of admix (wrong indices?).")
        if pop0 == pop1:
            raise ValueError("pop0 cannot equal pop1")
        if f < 0 or f > 1:
            raise ValueError("Admixture fraction must be between 0 and 1")

        Y_new = Numerics.admix(self, self.num_pops, pop0, pop1, f)
        if self.pop_ids is not None:
            new_pop_ids = self.pop_ids + [new_id]
        else:
            new_pop_ids = None
        return LDstats(Y_new, num_pops=self.num_pops + 1, pop_ids=new_pop_ids)

    def merge(self, pop0, pop1, f, new_id="Merged"):
        """
        Merger of populations pop0 and pop1, with fraction f from pop0
        and 1-f from pop1. Places new population at the end, then marginalizes
        pop0 and pop1. To admix two populations and keep one or both, use pulse
        migrate or admix, respectively.

        :param pop0: First population to merge.
        :type pop0: int
        :param pop1: Second population to merge.
        :type pop1: int
        :param f: The fraction of ancestry contributed by pop0, so pop1 contributes
            1 - f.
        :type f: float
        :param new_id: The name of the merged population.
        :type new_id: str, optional
        """
        Y_new = self.admix(pop0, pop1, f, new_id=new_id)
        Y_new = Y_new.marginalize([pop0, pop1])
        return Y_new

    def pulse_migrate(self, pop0, pop1, f):
        """
        Pulse migration/admixure event from pop0 to pop1, with fraction f
        replacement. We use the admix function above. We want to keep the
        original population names the same, if they are given in the LDstats
        object, so we use new_pop=self.pop_ids[pop1].

        We admix pop0 and pop1 with fraction f and 1-f, then swap the new
        admixed population with pop1, then marginalize the original pop1.

        :param pop0: The index of the source population.
        :type pop0: int
        :param pop1: The index of the target population.
        :type pop1: int
        :param f: The fraction of ancestry contributed by the source population.
        :type f: float
        """
        if (
            self.num_pops < 2
            or pop0 >= self.num_pops
            or pop1 >= self.num_pops
            or pop0 < 0
            or pop1 < 0
        ):
            raise ValueError("Improper usage of admix (wrong indices?).")
        if pop0 == pop1:
            raise ValueError("pop0 cannot equal pop1")
        if f < 0 or f > 1:
            raise ValueError("Admixture fraction must be between 0 and 1")

        if self.pop_ids is not None:
            Y_new = self.admix(pop0, pop1, f, new_id=self.pop_ids[pop1])
        else:
            Y_new = self.admix(pop0, pop1, f)
        Y_new = Y_new.swap_pops(pop1, Y_new.num_pops - 1)
        Y_new = Y_new.marginalize(Y_new.num_pops - 1)
        return Y_new

    # Make from_file a static method, so we can use it without an instance.
    @staticmethod
    def from_file(fid, return_statistics=False, return_comments=False):
        """
        Read LD statistics from file.

        :param fid: The file name to read from or an open file object.
        :type fid: str
        :param return_statistics: If true, returns statistics writen to file.
        :type return_statistics: bool, optional
        :param return_comments: If true, the return value is (y, comments), where
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

        # Read the num pops and pop_ids, if given
        line_spl = line.split()
        num_pops = int(line_spl[0])
        if len(line_spl) > 1:
            # get the pop_ids
            pop_ids = line_spl[1:]
            if num_pops != len(pop_ids):
                print("Warning: num_pops does not match number of pop_ids.")
        else:
            pop_ids = None

        # Get the statistic names
        ld_stats = fid.readline().split()
        het_stats = fid.readline().split()
        if ld_stats == ["ALL"]:
            ld_stats = Util.moment_names(num_pops)[0]
        if het_stats == ["ALL"]:
            het_stats = Util.moment_names(num_pops)[1]
        statistics = [ld_stats, het_stats]

        # Get the number of LD statistic rows and read LD data
        num_ld_rows = int(fid.readline().strip())
        data = []
        for r in range(num_ld_rows):
            data.append(numpy.fromstring(fid.readline().strip(), sep=" "))

        # Read heterozygosity data
        data.append(numpy.fromstring(fid.readline().strip(), sep=" "))

        # If we opened a new file, clean it up.
        if newfile:
            fid.close()

        y = LDstats(data, num_pops=num_pops, pop_ids=pop_ids)

        if return_statistics:
            if not return_comments:
                return y, statistics
            else:
                return y, statistics, comments
        else:
            if not return_comments:
                return y
            else:
                return y, comments

    def to_file(self, fid, precision=16, statistics="ALL", comment_lines=[]):
        """
        Write LD statistics to file.

        The file format is:

        - # Any number of comment lines beginning with a '#'
        - A single line containing an integer giving the number of
          populations.
        - On the *same line*, optional, the names of those populations. If
          names are given, there needs to be the same number of pop_ids
          as the integer number of populations. For example, the line could
          be '3 YRI CEU CHB'.
        - A single line giving the names of the *LD* statistics, in the order
          they appear for each recombination rate distance or bin.
          Optionally, this line could read ALL, indicating that every
          statistic in the basis is given, and in the 'correct' order.
        - A single line giving the names of the *heterozygosity* statistics,
          in the order they appear in the final row of data. Optionally,
          this line could read ALL.
        - A line giving the number of recombination rate bins/distances we
          have data for (so we know how many to read)
        - One line for each row of LD statistics.
        - A single line for the heterozygosity statistics.

        :param fid: The file name to write to or an open file object.
        :type fid: str
        :param precision: The precision with which to write out entries of the LD stats.
            (They are formated via %.<p>g, where <p> is the precision.)
        :type precision: int
        :param statistics: Defaults to 'ALL', meaning all statistics are given in the
            LDstats object. Otherwise, list of two lists, first giving
            present LD stats, and the second giving present het stats.
        :type statistics: list of list of strings
        :param comment_lines: List of strings to be used as comment lines in the header
            of the output file.
            I use comment lines mainly to record the recombination
            bins or distances given in the LDstats (something like
            "'edges = ' + str(r_edges)".
        :type comment_lines: list of srtings
        """

        # if statistics is ALL, check to make sure the lengths are correct
        if statistics != "ALL":
            ld_stat_names, het_stat_names = statistics
        else:
            ld_stat_names, het_stat_names = Util.moment_names(self.num_pops)

        all_correct_length = 1
        for ld_stats in self.LD():
            if len(ld_stats) != len(ld_stat_names):
                all_correct_length = 0
                break
        if len(self.H()) != len(het_stat_names):
            all_correct_length = 0

        if all_correct_length == 0:
            raise ValueError(
                "Length of data arrays does not match expected \
                              length of statistics."
            )

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

        # Write out the number of populations and pop_ids if given
        fid.write("%i" % self.num_pops)
        if self.pop_ids is not None:
            for pop in self.pop_ids:
                fid.write(" %s" % pop)
        fid.write(os.linesep)

        # Write out LD statistics
        if statistics == "ALL":
            fid.write(statistics)
        else:
            for stat in statistics[0]:
                fid.write("%s " % stat)
        fid.write(os.linesep)

        # Write out het statistics
        if statistics == "ALL":
            fid.write(statistics)
        else:
            for stat in statistics[1]:
                fid.write("%s " % stat)
        fid.write(os.linesep)

        # Write the LD data to the file
        fid.write("%i" % len(self.LD()))
        fid.write(os.linesep)
        for ld_stats in self.LD():
            for stat in ld_stats:
                fid.write("%%.%ig " % precision % stat)
            fid.write(os.linesep)

        # Write the het data to the file
        for stat in self.H():
            fid.write("%%.%ig " % precision % stat)

        # Close file
        if newfile:
            fid.close()

    @staticmethod
    def from_demes(
        g,
        sampled_demes,
        sample_times=None,
        rho=None,
        theta=0.001,
        r=None,
        u=None,
    ):
        """
        Takes a deme graph and computes the LD stats. ``demes`` is a package for
        specifying demographic models in a user-friendly, human-readable YAML
        format. This function automatically parses the demographic description
        and returns a LD for the specified populations and recombination and
        mutation rates.

        :param g: A ``demes`` DemeGraph from which to compute the LD.
        :type g: :class:`demes.DemeGraph`
        :param sampled_demes: A list of deme IDs to take samples from. We can repeat
            demes, as long as the sampling of repeated deme IDs occurs at distinct
            times.
        :type sampled_demes: list of strings
        :param sample_times: If None, assumes all sampling occurs at the end of the
            existence of the sampled deme. If there are
            ancient samples, ``sample_times`` must be a list of same length as
            ``sampled_demes``, giving the sampling times for each sampled
            deme. Sampling times are given in time units of the original deme graph,
            so might not necessarily be generations (e.g. if ``g.time_units`` is years)
        :type sample_times: list of floats, optional
        :param rho: The population-size scaled recombination rate(s). Can be None, a
            non-negative float, or a list of values.
        :param theta: The population-size scaled mutation rate. This defaults to
            ``theta=0.001``, which is very roughly what is observed in humans.
        :param r: The raw recombination rate. Can be None, a non-negative float, or a
            list of values.
        :param u: The raw per-base mutation rate. ``theta`` is set to ``4 * Ne * u``,
            where ``Ne`` is the reference population size from the root deme.
        :return: A ``moments.LD`` LD statistics object, with number of populations equal
            to the length of ``sampled_demes``.
        :rtype: :class:`moments.LD.LDstats`
        """

        if isinstance(g, str):
            dg = demes.load(g)
        else:
            dg = g

        y = moments.Demes.Demes.LD(
            dg,
            sampled_demes,
            sample_times=sample_times,
            rho=rho,
            theta=theta,
            r=r,
            u=u,
        )
        return y

    @staticmethod
    def steady_state(
        nus, m=None, rho=None, theta=0.001, selfing_rate=None, pop_ids=None
    ):
        """

        Computes the steady state solution for any number of populations. The
        number of populations is determined by the length of ``nus``, which is
        a list with relative population sizes (often, these will be set to 1,
        meaning sizes are equal to some reference or ancestral population
        size).

        If the solution for more than one population is desired, we must
        provide ``m``, an n-by-n migration matrix, and there must be migrations
        that connect all populations so that a steady state solution exists.
        This corresponds to an island model with potentially asymmetric
        migration and allows for unequal population sizes.

        :param nus: The relative population sizes, with one or two entries,
            corresponding to a steady state solution with one or two populations,
            resp.
        :type nus: list-like
        :param m: A migration matrix, only provided when the length of `nus` is 2
            or more.
        :type m: array-like
        :param rho: The population-size scaled recombination rate(s). Can be None, a
            non-negative float, or a list of values.
        :param theta: The population-size scaled mutation rate
        :type theta: float
        :param selfing_rate: Self-fertilization rate(s), given as a number (for a
            single population, or list of numbers (for two populations). Selfing
            rates must be between 0 and 1.
        :type selfing_rate: number or list of numbers
        :param pop_ids: The population IDs.
        :type pop_ids: list of strings
        :return: A ``moments.LD`` LD statistics object.
        :rtype: :class:`moments.LD.LDstats`
        """
        num_pops = len(nus)
        if num_pops == 1:
            if m is not None:
                raise ValueError(
                    "Migration matrix cannot be provided for one population."
                )
        elif num_pops >= 2:
            if m is None:
                raise ValueError(
                    "Migration matrix must be provided for the steady state solution "
                    "with two or more populations"
                )

        if pop_ids is not None and len(pop_ids) != num_pops:
            raise ValueError("pop_ids must be a list of length equal to nus.")

        data = Numerics.steady_state(
            nus, m=m, rho=rho, theta=theta, selfing_rate=selfing_rate
        )
        y = LDstats(data, num_pops=num_pops, pop_ids=pop_ids)
        return y

    # Ensures that when arithmetic is done with LDstats objects,
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
    if isinstance(other, np.ma.masked_array):
        newdata = self.%(method)s (other.data)
        newmask = np.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.%(method)s (other)
        newmask = self.mask
    outLDstats = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   num_pops=self.num_pops, pop_ids=self.pop_ids)
    return outLDstats
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
    if isinstance(other, np.ma.masked_array):
        self.%(method)s (other.data)
        self.mask = np.ma.mask_or(self.mask, other.mask)
    else:
        self.%(method)s (other)
    return self
"""
            % {"method": method}
        )

    def integrate(
        self,
        nu,
        tf,
        dt=None,
        dt_fac=0.02,
        rho=None,
        theta=0.001,
        m=None,
        selfing=None,
        selfing_rate=None,
        frozen=None,
    ):
        """
        Integrates the LD statistics forward in time. When integrating LD statistics
        for a list of recombination rates and mutation rate, they must be passed
        as keywork arguments to this function. We can integrate either single-population
        LD statistics up to order 10, or multi-population LD statistics but only
        for order 2 (which includes :math:`D^2`, :math:`Dz`, and :math:`\\pi_2`).

        :param nu: The relative population size, may be a function of time,
            given as a list [nu1, nu2, ...]
        :type nu: list or function
        :param tf: Total time to integrate
        :type tf: float
        :param dt: Integration timestep. This is deprecated! Use `dt_fac` instead.
        :type dt: float
        :param dt_fac: The integration time step factor, so that dt is determined by
            `tf * dt_fac`. Note: Should also build in adaptive time-stepping... this
            is to come.
        :type dt_fac: float
        :param rho: Can be a single recombination rate or list of recombination rates
            (in which case we are integrating a list of LD stats for each rate)
        :type rho: float or list of floats
        :param theta: The per base population-scaled mutation rate (4N*mu)
            if we pass [theta1, theta2], differing mutation rates at left and right
            locus, implemented in the ISM=True model
        :param m: The migration matrix (num_pops x num_pops, storing m_ij migration rates
            where m_ij is probability that a lineage in i had parent in j
            m_ii is unused, and found by summing off diag elements in the ith row
        :type m: array
        :param selfing: A list of selfing probabilities, same length as nu.
        :type selfing: list of floats
        :param selfing_rate: Alias for selfing.
        :type selfing_rate: list of floats
        :param frozen: A list of True and False same length as nu. True implies that a
            lineage is frozen (as in ancient samples). False integrates as normal.
        :type frozen: list of bools
        """
        num_pops = self.num_pops

        if tf == 0.0:
            return
        if tf < 0 or np.isinf(tf):
            raise ValueError("Integration time must be positive and finite.")

        if rho is None and len(self) > 1:
            raise ValueError("There are LD statistics, but rho is None.")
        elif rho is not None and np.isscalar(rho) and len(self) != 2:
            raise ValueError(
                "Single rho passed but LD object does have correct number of entries."
            )
        elif (
            rho is not None and np.isscalar(rho) == False and len(rho) != len(self) - 1
        ):
            raise ValueError("Mismatch length of input rho and size of LD object.")

        if rho is not None and np.isscalar(rho) == False and len(rho) == 1:
            rho = rho[0]

        if callable(nu):
            if len(nu(0)) != num_pops:
                raise ValueError("len of pop size function must equal number of pops.")
        else:
            if len(nu) != num_pops:
                raise ValueError("len of pop sizes must equal number of pops.")

        if m is not None and num_pops > 1:
            if np.shape(m) != (num_pops, num_pops):
                raise ValueError(
                    "migration matrix incorrectly defined for number of pops."
                )

        if frozen is not None:
            if len(frozen) != num_pops:
                raise ValueError("frozen must have same length as number of pops.")

        if selfing_rate is not None:
            if selfing is not None:
                raise ValueError("Cannot specify both selfing and selfing_rate")
            selfing = selfing_rate
        if selfing is not None:
            if len(selfing) != num_pops:
                raise ValueError("selfing must have same length as number of pops.")

        # enforce minimum 10 time steps per integration
        # if tf < dt * 10:
        #    dt_adj = tf / 10
        # else:
        #    dt_adj = dt * 1.0
        if dt is not None:
            warnings.warn(
                "dt is deprecated, use dt_fac to control integration time steps",
                DeprecationWarning,
            )
        if dt_fac <= 0 or dt_fac > 1:
            raise ValueError("dt_fac must be between 0 and 1")

        self[:] = Numerics.integrate(
            self[:],
            nu,
            tf,
            dt=None,
            dt_fac=dt_fac,
            rho=rho,
            theta=theta,
            m=m,
            num_pops=num_pops,
            selfing=selfing,
            frozen=frozen,
        )


# Allow LDstats objects to be pickled.
try:
    import copy_reg
except:
    import copyreg


def LDstats_pickler(y):
    return LDstats_unpickler, (y[:], y.num_pops, y.pop_ids)


def LDstats_unpickler(data, num_pops, pop_ids):
    return LDstats(data[:], num_pops=num_pops, pop_ids=pop_ids)


try:
    copy_reg.pickle(LDstats, LDstats_pickler, LDstats_unpickler)
except:
    copyreg.pickle(LDstats, LDstats_pickler, LDstats_unpickler)
