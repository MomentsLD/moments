_imported_h5py = 0
_imported_allel = 0
_imported_pandas = 0
import os
from datetime import datetime


def current_time():
    return " [" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"


try:
    os.environ["NUMEXPR_MAX_THREADS"] = "272"
    import allel

    _imported_allel = 1
except ImportError:
    pass

try:
    import h5py

    _imported_h5py = 1
except ImportError:
    pass

try:
    import pandas

    _imported_pandas = 1
except ImportError:
    pass

from . import Util


def check_imports():
    if _imported_allel == 0:
        raise ImportError("Failed to import scikit-allel which is needed for Parsing")
    if _imported_h5py == 0:
        raise ImportError("Failed to import h5py which is needed for Parsing")
    if _imported_pandas == 0:
        raise ImportError("Failed to import pandas which is needed for Parsing")


import numpy as np
from collections import Counter, defaultdict

# from . import stats_from_genotype_counts as sgc
from . import stats_from_haplotype_counts as shc
import sys
import itertools

ld_extensions = 0
try:
    import genotype_calculations as gcs
    import genotype_calculations_multipop as gcs_mp
    import sparse_tallying as spt

    ld_extensions = 1
except ImportError:
    pass

# turn off UserWarnings from allel
import warnings

if _imported_allel:
    warnings.filterwarnings(
        action="ignore", message="'GQ' FORMAT", category=UserWarning
    )


def _load_h5(vcf_file, report=True):
    check_imports()
    ## open the h5 callset, create if doesn't exist
    ## note that if the h5 file exists, but isn't properly written,
    ## you will need to delete and recreate
    ## saves h5 callset as same name and path,
    ## but with h5 extension instead of vcf or vcf.gz
    h5_file_path = vcf_file.split(".vcf")[0] + ".h5"
    try:
        callset = h5py.File(h5_file_path, mode="r")
    except (OSError, IOError):  # IOError merged into OSError in python 3
        if report is True:
            print(current_time(), "creating and saving h5 file")
            sys.stdout.flush()
        allel.vcf_to_hdf5(
            vcf_file,
            h5_file_path,
            fields="*",
            exclude_fields=["calldata/GQ"],
            overwrite=True,
        )
        callset = h5py.File(h5_file_path, mode="r")
    return callset


def get_genotypes(
    vcf_file, bed_file=None, chromosome=None, min_bp=None, use_h5=True, report=True
):
    """
    Given a vcf file, we extract the biallelic SNP genotypes. If bed_file is None,
    we use all valid variants. Otherwise we filter genotypes by the given bed file.
    If chromosome is given, filters to keep snps only in given chrom (useful for vcfs
    spanning multiple chromosomes).

    If use_h5 is True, we try to load the h5 file, which has the same path/name as
    vcf_file, but with {fname}.h5 instead of {fname}.vcf or {fname}.vcf.gz. If the h5
    file does not exist, we create it and save it as {fname}.h5

    Returns (biallelic positions, biallelic genotypes, biallelic allele counts,
    sampled ids).

    :param vcf_file: A VCF-formatted file.
    :type vcf_file: str
    :param bed_file: A bed file specifying regions to compute statistics from. The
        chromosome name formatting must match the chromosome name formatting of the
        input VCF (i.e., both carry the leading "chr" or both omit it).
    :type bed_file: str, optional
    :param min_bp: only used with bed file, filters out features that are smaller
        than ``min_bp``.
    :type min_bp: int, optional
    :param chromosome: Chromosome to compute LD statistics from.
    :type chromosome: int or str, optional
    :param use_h5: If use_h5 is True, we try to load the h5 file, which has the
        same path/name as vcf_file, but with .h5 instead of .vcf or .vcf.gz extension.
        If the h5 file does not exist, we create it and save it with .h5 extension.
        Defaults to True.
    :type use_h5: bool, optional
    :param report: Prints progress updates if True, silent otherwise. Defaults to True.
    :type report: bool, optional
    """

    check_imports()

    if use_h5 is True:
        callset = _load_h5(vcf_file, report=report)
    else:
        callset = allel.read_vcf(vcf_file)

    all_genotypes = allel.GenotypeChunkedArray(callset["calldata/GT"])
    all_positions = callset["variants/POS"][:]

    # if there are multiple chromosomes in the VCF, we require a specified chromosome
    try:
        all_chromosomes = np.array([c.decode() for c in callset["variants/CHROM"][:]])
    except AttributeError:
        all_chromosomes = callset["variants/CHROM"][:]

    num_chromosomes = len(set(all_chromosomes))
    if num_chromosomes == 1:
        if chromosome is None:
            chromosome = all_chromosomes[0]
    elif num_chromosomes > 1:
        if chromosome is None:
            raise ValueError(
                "The input VCF has more than one chromosome present. "
                "The `chromosome` must be specified."
            )

    if str(chromosome) not in all_chromosomes:
        raise ValueError(
            f"The specified chromosome, {chromosome}, was not found among "
            "sites in the VCF. Double check the input chromosome name."
        )

    if num_chromosomes > 1:
        in_chromosome = all_chromosomes == str(chromosome)
        all_positions = all_positions.compress(in_chromosome)
        all_genotypes = all_genotypes.compress(in_chromosome)

    if bed_file is not None:
        bed_file_data = pandas.read_csv(bed_file, sep=r"\s+", header=None)
        bed_chromosomes = np.array(bed_file_data[0])
        bed_lefts = np.array(bed_file_data[1])
        bed_rights = np.array(bed_file_data[2])

        # only keep rows of bed file that match our chromosome
        bed_chrom_filter = bed_chromosomes.astype(str) == str(chromosome)
        bed_chromosomes = bed_chromosomes.compress(bed_chrom_filter)
        if len(bed_chromosomes) == 0:
            raise ValueError(
                "No regions of the bed file matched the input chromosome. "
                "Check that chromosome names match between bed file and VCF."
            )
        bed_lefts = bed_lefts.compress(bed_chrom_filter)
        bed_rights = bed_rights.compress(bed_chrom_filter)

        # if a minimum length of feature is specified, only keep long enough features
        if min_bp is not None:
            bp_filter = bed_rights - bed_lefts >= min_bp
            bed_lefts = bed_lefts.compress(bp_filter)
            bed_rights = bed_rights.compress(bp_filter)
            if len(bed_lefts) == 0:
                raise ValueError(
                    "No features in bed file were at least {min_bp} in length."
                )

        in_bed = all_positions < 0
        for left, right in zip(bed_lefts, bed_rights):
            in_bed = np.logical_or(
                in_bed, np.logical_and(all_positions >= left, all_positions < right)
            )

        all_positions = all_positions.compress(in_bed)
        all_genotypes = all_genotypes.compress(in_bed)

    all_genotypes_012 = all_genotypes.to_n_alt(fill=-1)

    # count alleles and only keep biallelic positions
    allele_counts = all_genotypes.count_alleles()
    is_biallelic = allele_counts.is_biallelic_01()
    biallelic_positions = all_positions.compress(is_biallelic)

    biallelic_genotypes_012 = all_genotypes_012.compress(is_biallelic)

    biallelic_allele_counts = allele_counts.compress(is_biallelic)
    biallelic_genotypes = all_genotypes.compress(is_biallelic)

    relevant_column = np.array([False] * biallelic_allele_counts.shape[1])
    relevant_column[0:2] = True
    biallelic_allele_counts = biallelic_allele_counts.compress(relevant_column, axis=1)

    # protect against variable encoding of sample id strings
    try:
        sample_ids = np.array([sid.decode() for sid in callset["samples"]])
    except AttributeError:
        sample_ids = callset["samples"]

    return biallelic_positions, biallelic_genotypes, biallelic_allele_counts, sample_ids


def _assign_r_pos(positions, rec_map):
    rs = np.zeros(len(positions))
    for ii, pos in enumerate(positions):
        if pos in np.array(rec_map[0]):
            rs[ii] = np.asarray(rec_map[1])[np.argwhere(pos == np.array(rec_map[0]))[0]]
        else:
            ## for now, if outside rec map, assign to nearest point,
            ## but later want to drop these positions
            if pos < rec_map[0].iloc[0]:
                rs[ii] = rec_map[1].iloc[0]
            elif pos > rec_map[0].iloc[-1]:
                rs[ii] = rec_map[1].iloc[-1]
            else:
                map_ii = np.where(pos >= np.array(rec_map[0]))[0][-1]
                l = rec_map[0][map_ii]
                r = rec_map[0][map_ii + 1]
                v_l = rec_map[1][map_ii]
                v_r = rec_map[1][map_ii + 1]
                rs[ii] = v_l + (v_r - v_l) * (pos - l) / (r - l)
    return rs


def _assign_recombination_rates(
    positions, map_file, map_name=None, map_sep=None, cM=True, report=True
):
    ## map_sep is now deprecated. we split by any white space
    if map_file == None:
        raise ValueError(
            "Need to pass a recombination map file. Otherwise can bin by physical distance."
        )
        sys.stdout.flush()
    try:
        rec_map = pandas.read_csv(map_file, sep=r"\s+")
    except:
        raise ValueError("Error loading recombination map.")
        sys.stdout.flush()

    if "Position(bp)" in rec_map.keys():
        map_positions = rec_map["Position(bp)"]
    else:
        map_positions = rec_map[rec_map.keys()[0]]
    if map_name is None:  # we use the first map column
        if report is True:
            print(current_time(), "No recombination map name given, trying Map(cM).")
            sys.stdout.flush()
        map_name = "Map(cM)"
    try:
        map_values = rec_map[map_name]
    except KeyError:
        print(
            current_time(),
            f"WARNING: map_name did not match {map_name}"
            " in recombination map file. Using the first column.",
        )
        map_values = rec_map[rec_map.keys()[1]]

    # for positions sticking out the end of the map,
    # they take the value of the closest position
    # ideally, you'd filter these out

    if cM == True:
        map_values /= 100

    pos_rs = _assign_r_pos(positions, [map_positions, map_values])

    return pos_rs


## We store a Genotype matrix, which has size L x n, in sparse format
## Indexed by position in Genotype array (0...L-1)
## and G_dict[i] = {1: J, 2: K}
## where J and K are sets of the diploid individual indices that
## have genotypes 1 or 2
## If there is any missing data, we also store set of individuals with -1's


def _sparsify_genotype_matrix(G):
    G_dict = {}
    if np.any(G == -1):
        missing = True
    else:
        missing = False
    for i in range(len(G)):
        G_dict[i] = {
            1: set(np.where(G[i, :] == 1)[0]),
            2: set(np.where(G[i, :] == 2)[0]),
        }
        if missing == True:
            G_dict[i][-1] = set(np.where(G[i, :] == -1)[0])
        else:
            G_dict[i][-1] = set()
    return G_dict, missing


def _sparsify_haplotype_matrix(G):
    G_dict = {}
    if np.any(G == -1):
        missing = True
    else:
        missing = False
    for i in range(len(G)):
        G_dict[i] = {
            1: set(np.where(G[i, :] == 1)[0]),
        }
        if missing == True:
            G_dict[i][-1] = set(np.where(G[i, :] == -1)[0])
        else:
            G_dict[i][-1] = set()
    return G_dict, missing


def _check_valid_genotype_matrix(G, genotypes):
    if genotypes:
        # 0, 1, 2 for genotype values, -1 for missing data
        if not np.all(
            np.logical_or(np.logical_or(np.logical_or(G == 0, G == 1), G == 2), G == -1)
        ):
            raise ValueError(
                "Genotype matrix must have values of 0, 1, or 2, or -1 for missing data"
            )
    else:
        # haplotypes: 0, 1, -1 for missing data
        if not np.all(np.logical_or(np.logical_or(G == 0, G == 1), G == -1)):
            raise ValueError(
                "Haplotype matrix must have values of 0 or 1, or -1 for missing data"
            )

    L, n = np.shape(G)
    if L > 46340:
        raise ValueError("Genotype matrix is too large, consider parallelizing LD calc")


def compute_pairwise_stats(Gs, genotypes=True):
    """
    Computes :math:`D^2`, :math:`Dz`, :math:`\\pi_2`, and :math:`D` for every
    pair of loci within a block of SNPs, coded as a genotype matrix.

    :param Gs: A genotype matrix, of size L-by-n, where
        L is the number of loci and n is the sample size.
        Missing data is encoded as -1.
    :param genotypes: If True, use 0, 1, 2 genotypes. If False,
        use 0, 1 phased haplotypes.
    """
    if ld_extensions != 1:
        raise ValueError(
            "Need to build LD cython extensions. "
            "Install moments with the flag `--ld_extensions`"
        )

    _check_valid_genotype_matrix(Gs, genotypes)

    L, n = np.shape(Gs)

    if genotypes:
        G_dict, any_missing = _sparsify_genotype_matrix(Gs)
        Counts = spt.count_genotypes_sparse(G_dict, n, missing=any_missing)
    else:
        G_dict, any_missing = _sparsify_haplotype_matrix(Gs)
        Counts = spt.count_haplotypes_sparse(G_dict, n, missing=any_missing)

    if genotypes:
        D = gcs.compute_D(Counts)
        D2 = gcs.compute_D2(Counts)
        Dz = gcs.compute_Dz(Counts)
        pi2 = gcs.compute_pi2(Counts)
    else:
        D = shc.D(Counts.T)
        D2 = shc.DD([Counts.T], [0, 0])
        Dz = shc.Dz([Counts.T], [0, 0, 0])
        pi2 = shc.pi2([Counts.T], [0, 0, 0, 0])

    return D2, Dz, pi2, D


def compute_average_stats(Gs, genotypes=True):
    """
    Takes the outputs of ``compute_pairwise_stats`` and returns
    the average value for each statistic.

    :param Gs: A genotype matrix, of size L-by-n, where
        L is the number of loci and n is the sample size.
        Missing data is encoded as -1.
    :param genotypes: If True, use 0, 1, 2 genotypes. If False,
        use 0, 1 phased haplotypes.
    """
    D2, Dz, pi2, D = compute_pairwise_stats(Gs, genotypes=True)
    return np.mean(D2), np.mean(Dz), np.mean(pi2), np.mean(D)


def compute_pairwise_stats_between(Gs1, Gs2, genotypes=True):
    """
    Computes :math:`D^2`, :math:`Dz`, :math:`\\pi_2`, and :math:`D`
    for every pair of loci between two blocks of SNPs, coded as
    genotype matrices.

    The Gs are matrices, where rows correspond to loci and columns to individuals.
    Both matrices must have the same number of individuals. If Gs1 has length L1
    and Gs2 has length L2, we compute all pairwise counts, which has size (L1*L2, 9).

    We use the sparse genotype matrix representation, where
    we first "sparsify" the genotype matrix, and then count
    two-locus genotype configurations from that, from which
    we compute two-locus statistics

    :param Gs1: A genotype matrices, of size L1 by n, where
        L1 is the number of loci and n is the sample size.
        Missing data is encoded as -1.
    :param Gs2: A genotype matrices, of size L2 by n, where
        L1 is the number of loci and n is the sample size.
        Missing data is encoded as -1.
    :param genotypes: If True, use 0, 1, 2 genotypes. If False,
        use 0, 1 phased haplotypes.
    """
    assert (
        ld_extensions == 1
    ), "Need to build LD cython extensions. Install moments with the flag `--ld_extensions`"

    _check_valid_genotype_matrix(Gs1, genotypes)
    _check_valid_genotype_matrix(Gs2, genotypes)

    L1, n1 = np.shape(Gs1)
    L2, n2 = np.shape(Gs1)

    if n1 != n2:
        raise ValueError("data must have same number of sequenced individuals")
    else:
        n = n1

    if genotypes:
        G_dict1, any_missing1 = _sparsify_genotype_matrix(Gs1)
        G_dict2, any_missing2 = _sparsify_genotype_matrix(Gs2)
    else:
        G_dict1, any_missing1 = _sparsify_haplotype_matrix(Gs1)
        G_dict2, any_missing2 = _sparsify_haplotype_matrix(Gs2)

    any_missing = np.logical_or(any_missing1, any_missing2)

    if genotypes:
        Counts = spt.count_genotypes_between_sparse(
            G_dict1, G_dict2, n, missing=any_missing
        )
    else:
        Counts = spt.count_haplotypes_between_sparse(
            G_dict1, G_dict2, n, missing=any_missing
        )

    if genotypes:
        D2 = gcs.compute_D2(Counts)
        Dz = gcs.compute_Dz(Counts)
        pi2 = gcs.compute_pi2(Counts)
        D = gcs.compute_D(Counts)
    else:
        D = shc.D(Counts.T)
        D2 = shc.DD([Counts.T], [0, 0])
        Dz = shc.Dz([Counts.T], [0, 0, 0])
        pi2 = shc.pi2([Counts.T], [0, 0, 0, 0])

    return D2, Dz, pi2, D


def compute_average_stats_between(Gs1, Gs2, genotypes=True):
    """
    Takes the outputs of compute_pairwise_stats_between and returns
    the average value for each statistic.

    :param Gs1: A genotype matrices, of size L1 by n, where
        L1 is the number of loci and n is the sample size.
        Missing data is encoded as -1.
    :param Gs2: A genotype matrices, of size L2 by n, where
        L1 is the number of loci and n is the sample size.
        Missing data is encoded as -1.
    """
    D2, Dz, pi2, D = compute_pairwise_stats_between(Gs1, Gs2, genotypes=genotypes)
    return np.mean(D2), np.mean(Dz), np.mean(pi2), np.mean(D)


def _count_types_sparse(
    genotypes,
    bins,
    sample_ids,
    positions=None,
    pos_rs=None,
    pop_file=None,
    pops=None,
    use_genotypes=True,
    report=True,
    report_spacing=1000,
    use_cache=True,
    stats_to_compute=None,
    normalized_by=None,
):
    assert (
        ld_extensions == 1
    ), "Need to build LD cython extensions. Install moments with the flag `--ld_extensions`"

    pop_indexes = {}
    if pops is not None:
        ## get columns to keep, and compress data and sample_ids
        samples = pandas.read_csv(pop_file, sep=r"\s+")
        cols_to_keep = np.array([False] * np.shape(genotypes)[1])
        all_samples_to_keep = []
        for pop in pops:
            all_samples_to_keep += list(samples[samples["pop"] == pop]["sample"])

        sample_list = list(sample_ids)
        for s in all_samples_to_keep:
            cols_to_keep[sample_list.index(s)] = True

        genotypes_pops = genotypes.compress(cols_to_keep, axis=1)
        sample_ids_pops = list(np.array(sample_list).compress(cols_to_keep))

        ## keep only biallelic genotypes from populations in pops, discard the rest
        allele_counts_pops = genotypes_pops.count_alleles()
        is_biallelic = allele_counts_pops.is_biallelic_01()
        genotypes_pops = genotypes_pops.compress(is_biallelic)

        ## for each population, get the indexes for each population
        for pop in pops:
            pop_indexes[pop] = np.array([False] * np.shape(genotypes_pops)[1])
            for s in samples[samples["pop"] == pop]["sample"]:
                pop_indexes[pop][sample_ids_pops.index(s)] = True
            if not np.any(pop_indexes[pop]):
                raise ValueError(f"population {pop} has no samples in data")

        if use_genotypes == False:
            pop_indexes_haps = {}
            for pop in pops:
                pop_indexes_haps[pop] = np.reshape(
                    list(zip(pop_indexes[pop], pop_indexes[pop])),
                    (2 * len(pop_indexes[pop]),),
                )

        if positions is not None:
            positions = positions.compress(is_biallelic)
        if pos_rs is not None:
            pos_rs = pos_rs.compress(is_biallelic)

    else:
        if report == True:
            print(
                current_time(),
                "No populations given, using all samples as one population.",
            )
            sys.stdout.flush()
        pops = ["ALL"]
        pop_indexes["ALL"] = np.array([True] * np.shape(genotypes)[1])
        genotypes_pops = genotypes
        if use_genotypes == False:
            pop_indexes_haps = {}
            for pop in pops:
                pop_indexes_haps[pop] = np.reshape(
                    list(zip(pop_indexes[pop], pop_indexes[pop])),
                    (2 * len(pop_indexes[pop]),),
                )

    ## convert to 0,1,2 format
    if use_genotypes == True:
        genotypes_pops_012 = genotypes_pops.to_n_alt()
    else:
        try:
            haplotypes_pops_01 = genotypes_pops.to_haplotypes()
        except AttributeError:
            print(
                current_time(),
                "warning: attempted to get haplotypes from phased genotypes, returned attribute error. Using input as haplotypes.",
            )
            haplotypes_pops_01 = genotypes_pops

    if pos_rs is not None:
        rs = pos_rs
    elif positions is not None:
        rs = positions

    ns = {}
    for pop in pops:
        if use_genotypes == True:
            ns[pop] = sum(pop_indexes[pop])
        else:
            ns[pop] = 2 * sum(pop_indexes[pop])

    bins = np.array(bins)

    ## split and sparsify the geno/haplo-type arrays for each population
    if use_genotypes == True:
        genotypes_by_pop = {}
        any_missing = False
        for pop in pops:
            temp_genotypes = genotypes_pops_012.compress(pop_indexes[pop], axis=1)
            genotypes_by_pop[pop], this_missing = _sparsify_genotype_matrix(
                temp_genotypes
            )
            any_missing = np.logical_or(any_missing, this_missing)
    else:
        haplotypes_by_pop = {}
        any_missing = False
        for pop in pops:
            temp_haplotypes = haplotypes_pops_01.compress(pop_indexes_haps[pop], axis=1)
            haplotypes_by_pop[pop], this_missing = _sparsify_haplotype_matrix(
                temp_haplotypes
            )
            any_missing = np.logical_or(any_missing, this_missing)

    #    if use_cache == True:
    #        run loop that computes type_counts cache
    #    else
    #        run loop that adds to sums

    ## if use_cache is True, type_counts will store the number of times we
    ## see each genotype count configuration within each bin
    ## if use_cache is False, we add to the running total of sums of each
    ## statistic as we count their genotypes, never storing the counts of configurations
    bs = list(zip(bins[:-1], bins[1:]))
    if use_cache == True:
        type_counts = {}
        for b in bs:
            type_counts[b] = defaultdict(int)
    else:
        sums = {}
        for b in bs:
            sums[b] = {}
            for stat in stats_to_compute[0]:
                sums[b][stat] = 0

    ## loop through left positions and pair with positions to the right
    ## within the bin windows
    ## this is a very inefficient, naive approach
    for ii, r in enumerate(rs[:-1]):
        if report is True:
            if ii % report_spacing == 0:
                print(
                    current_time(),
                    "tallied two locus counts {0} of {1} positions".format(ii, len(rs)),
                )
                sys.stdout.flush()

        ## loop through each bin, picking out the positions to the right
        ## of the left locus that fall within the given bin
        if pos_rs is not None:
            distances = pos_rs - r
        else:
            distances = positions - r

        filt = np.logical_and(
            np.logical_and(distances >= bs[0][0], distances < bs[-1][1]),
            positions != positions[ii],
        )
        filt[ii] = False  # don't compare to mutations at same base pair position
        right_indices = np.where(filt == True)[0]

        ## if there are no variants within the bin's distance to the right,
        ## continue to next bin
        if len(right_indices) == 0:
            continue

        right_start = right_indices[0]
        right_end = right_indices[-1] + 1

        if use_cache == False:
            counts_ii = {}
            for b in bs:
                counts_ii[b] = [[] for pop_ind in range(len(pops))]

        ## loop through right loci and count two-locus genotypes
        for jj in range(right_start, right_end):
            # get the bin that this pair belongs to
            r_dist = distances[jj]
            bin_ind = np.where(r_dist >= bins)[0][-1]
            b = bs[bin_ind]

            # count genotypes within each population
            if use_genotypes == True:
                cs = tuple(
                    [
                        spt.tally_sparse(
                            genotypes_by_pop[pop][ii],
                            genotypes_by_pop[pop][jj],
                            ns[pop],
                            any_missing,
                        )
                        for pop in pops
                    ]
                )
            else:
                cs = tuple(
                    [
                        spt.tally_sparse_haplotypes(
                            haplotypes_by_pop[pop][ii],
                            haplotypes_by_pop[pop][jj],
                            ns[pop],
                            any_missing,
                        )
                        for pop in pops
                    ]
                )

            if use_cache == True:
                type_counts[b][cs] += 1
            else:
                for pop_ind in range(len(pops)):
                    counts_ii[b][pop_ind].append(cs[pop_ind])

        if use_cache == False:
            for b in bs:
                these_counts = np.array(counts_ii[b])
                if these_counts.shape[1] == 0:
                    continue
                for stat in stats_to_compute[0]:
                    sums[b][stat] += _call_sgc(
                        stat, these_counts.swapaxes(1, 2), use_genotypes
                    ).sum()

    if use_cache == True:
        return type_counts
    else:
        return sums


def _call_sgc(stat, Cs, use_genotypes=True):
    """
    stat = 'DD', 'Dz', or 'pi2', with underscore indices (like 'DD_1_1')
    Cs = L \times n array, L number of count configurations, n = 4 or 9
    (for haplotypes or genotypes)
    """
    assert (
        ld_extensions == 1
    ), "Need to build LD cython extensions. Install moments with the flag `--ld_extensions`"

    s = stat.split("_")[0]
    pop_nums = [int(p) for p in stat.split("_")[1:]]

    if Cs.__class__ != np.ndarray:
        raise ValueError("Cs expected to be a numpy array, got", Cs.__class__)
    if Cs.dtype != np.int64:
        Cs = Cs.astype(np.int64)

    if s == "DD":
        if use_genotypes == True:
            return gcs_mp.DD(Cs, pop_nums)
        else:
            return shc.DD(Cs, pop_nums)
    if s == "Dz":
        ii, jj, kk = pop_nums
        if jj == kk:
            if use_genotypes == True:
                return gcs_mp.Dz(Cs, pop_nums)
            else:
                return shc.Dz(Cs, pop_nums)
        else:
            if use_genotypes == True:
                return 1.0 / 2 * gcs_mp.Dz(Cs, [ii, jj, kk]) + 1.0 / 2 * gcs_mp.Dz(
                    Cs, [ii, kk, jj]
                )
            else:
                return 1.0 / 2 * shc.Dz(Cs, [ii, jj, kk]) + 1.0 / 2 * shc.Dz(
                    Cs, [ii, kk, jj]
                )
    if s == "pi2":
        (
            ii,
            jj,
            kk,
            ll,
        ) = pop_nums  ### this doesn't consider the symmetry between p/q yet...
        if ii == jj:
            if kk == ll:
                if ii == kk:  # all the same
                    if use_genotypes == True:
                        return gcs_mp.pi2(Cs, [ii, jj, kk, ll])
                    else:
                        return shc.pi2(Cs, [ii, jj, kk, ll])
                else:  # (1, 1; 2, 2)
                    if use_genotypes == True:
                        return (
                            1.0
                            / 2
                            * (
                                gcs_mp.pi2(Cs, [ii, jj, kk, ll])
                                + gcs_mp.pi2(Cs, [kk, ll, ii, jj])
                            )
                        )
                    else:
                        return (
                            1.0
                            / 2
                            * (
                                shc.pi2(Cs, [ii, jj, kk, ll])
                                + shc.pi2(Cs, [kk, ll, ii, jj])
                            )
                        )
            else:  # (1, 1; 2, 3) or (1, 1; 1, 2)
                if use_genotypes == True:
                    return (
                        1.0
                        / 4
                        * (
                            gcs_mp.pi2(Cs, [ii, jj, kk, ll])
                            + gcs_mp.pi2(Cs, [ii, jj, ll, kk])
                            + gcs_mp.pi2(Cs, [kk, ll, ii, jj])
                            + gcs_mp.pi2(Cs, [ll, kk, ii, jj])
                        )
                    )
                else:
                    return (
                        1.0
                        / 4
                        * (
                            shc.pi2(Cs, [ii, jj, kk, ll])
                            + shc.pi2(Cs, [ii, jj, ll, kk])
                            + shc.pi2(Cs, [kk, ll, ii, jj])
                            + shc.pi2(Cs, [ll, kk, ii, jj])
                        )
                    )
        else:
            if kk == ll:  # (1, 2; 3, 3) or (1, 2; 2, 2)
                if use_genotypes == True:
                    return (
                        1.0
                        / 4
                        * (
                            gcs_mp.pi2(Cs, [ii, jj, kk, ll])
                            + gcs_mp.pi2(Cs, [jj, ii, kk, ll])
                            + gcs_mp.pi2(Cs, [kk, ll, ii, jj])
                            + gcs_mp.pi2(Cs, [kk, ll, jj, ii])
                        )
                    )
                else:
                    return (
                        1.0
                        / 4
                        * (
                            shc.pi2(Cs, [ii, jj, kk, ll])
                            + shc.pi2(Cs, [jj, ii, kk, ll])
                            + shc.pi2(Cs, [kk, ll, ii, jj])
                            + shc.pi2(Cs, [kk, ll, jj, ii])
                        )
                    )
            else:  # (1, 2; 3, 4)
                if use_genotypes == True:
                    return (
                        1.0
                        / 8
                        * (
                            gcs_mp.pi2(Cs, [ii, jj, kk, ll])
                            + gcs_mp.pi2(Cs, [ii, jj, ll, kk])
                            + gcs_mp.pi2(Cs, [jj, ii, kk, ll])
                            + gcs_mp.pi2(Cs, [jj, ii, ll, kk])
                            + gcs_mp.pi2(Cs, [kk, ll, ii, jj])
                            + gcs_mp.pi2(Cs, [ll, kk, ii, jj])
                            + gcs_mp.pi2(Cs, [kk, ll, jj, ii])
                            + gcs_mp.pi2(Cs, [ll, kk, jj, ii])
                        )
                    )
                else:
                    return (
                        1.0
                        / 8
                        * (
                            shc.pi2(Cs, [ii, jj, kk, ll])
                            + shc.pi2(Cs, [ii, jj, ll, kk])
                            + shc.pi2(Cs, [jj, ii, kk, ll])
                            + shc.pi2(Cs, [jj, ii, ll, kk])
                            + shc.pi2(Cs, [kk, ll, ii, jj])
                            + shc.pi2(Cs, [ll, kk, ii, jj])
                            + shc.pi2(Cs, [kk, ll, jj, ii])
                            + shc.pi2(Cs, [ll, kk, jj, ii])
                        )
                    )


def _cache_ld_statistics(type_counts, ld_stats, bins, use_genotypes=True, report=True):
    ### This function might not be needed anymore if we completely remove caching
    bs = list(zip(bins[:-1], bins[1:]))

    estimates = {}
    for b in bs:
        for cs in type_counts[b].keys():
            estimates.setdefault(cs, {})

    all_counts = np.array(list(estimates.keys()))
    all_counts = np.swapaxes(all_counts, 0, 1)
    all_counts = np.swapaxes(all_counts, 1, 2)

    for stat in ld_stats:
        if report is True:
            print(current_time(), "computing " + stat)
            sys.stdout.flush()
        vals = _call_sgc(stat, all_counts, use_genotypes)
        for ii in range(len(all_counts[0, 0])):
            cs = all_counts[:, :, ii]
            estimates[tuple([tuple(c) for c in cs])][stat] = vals[ii]
    return estimates


def _get_ld_stat_sums(type_counts, ld_stats, bins, use_genotypes=True, report=True):
    """
    return sums[b][stat]
    """
    ### this is super inefficient, just trying to get around memory issues

    bs = list(zip(bins[:-1], bins[1:]))
    sums = {}
    if use_genotypes is True:
        empty_genotypes = tuple([0] * 9)
    else:
        empty_genotypes = tuple([0] * 4)

    for stat in ld_stats:
        if report is True:
            print(current_time(), "computing " + stat)
            sys.stdout.flush()
        # set counts of non-used stats to zeros, then take set
        pops_in_stat = sorted(list(set(int(p) for p in stat.split("_")[1:])))
        stat_counts = {}
        for b in bs:
            for cs in type_counts[b].keys():
                this_count = list(cs)
                for i in range(len(cs)):
                    if i not in pops_in_stat:
                        this_count[i] = empty_genotypes
                stat_counts.setdefault(tuple(this_count), defaultdict(int))
                stat_counts[tuple(this_count)][b] += type_counts[b][cs]

        all_counts = np.array(list(stat_counts.keys()))
        all_counts = np.swapaxes(all_counts, 0, 1)
        all_counts = np.swapaxes(all_counts, 1, 2)
        vals = _call_sgc(stat, all_counts, use_genotypes)

        estimates = {}
        for v, ii in zip(vals, range(len(all_counts[0, 0]))):
            cs = tuple(tuple(c) for c in all_counts[:, :, ii])
            estimates[cs] = v

        for b in bs:
            sums.setdefault(b, {})
            sums[b][stat] = 0
            for cs in stat_counts:
                sums[b][stat] += stat_counts[cs][b] * estimates[cs]

    return sums


def _get_H_statistics(
    genotypes, sample_ids, pop_file=None, pops=None, ac_filter=True, report=True
):
    """
    Het values are not normalized by sequence length,
    would need to compute L from bed file.
    """

    if pop_file == None and pops == None:
        if report == True:
            print(
                current_time(),
                "No population file or population names given, "
                "assuming all samples as single pop.",
            )
            sys.stdout.flush()
    elif pops == None:
        raise ValueError("pop_file given, but not population names...")
        sys.stdout.flush()
    elif pop_file == None:
        raise ValueError("Population names given, but not pop_file...")
        sys.stdout.flush()

    if pops == None:
        pops = ["ALL"]

    if pop_file is not None:
        samples = pandas.read_csv(pop_file, sep=r"\s+")
        populations = np.array(samples["pop"].value_counts().keys())
        samples.reset_index(drop=True, inplace=True)

        ### should use this above when counting two locus genotypes
        sample_ids_list = list(sample_ids)
        subpops = {
            # for each population, get the list of samples that
            # belong to the population
            pop_iter: [
                sample_ids_list.index(ind)
                for ind in samples[samples["pop"] == pop_iter]["sample"]
            ]
            for pop_iter in pops
        }

        ac_subpop = genotypes.count_alleles_subpops(subpops)
    else:
        subpops = {pop_iter: list(range(len(sample_ids))) for pop_iter in pops}
        ac_subpop = genotypes.count_alleles_subpops(subpops)

    # ensure at least 2 allele counts per pop
    min_ac_filter = [True] * len(ac_subpop)
    if ac_filter == True:
        for pop in pops:
            min_ac_filter = np.logical_and(
                min_ac_filter, np.sum(ac_subpop[pop], axis=1) >= 2
            )
    ac_subpop_filt = {}
    for pop in pops:
        ac_subpop_filt[pop] = np.asarray(ac_subpop[pop]).compress(min_ac_filter, axis=0)

    Hs = {}
    for ii, pop1 in enumerate(pops):
        for pop2 in pops[ii:]:
            if pop1 == pop2:
                H = np.sum(
                    2.0
                    * ac_subpop_filt[pop1][:, 0]
                    * ac_subpop_filt[pop1][:, 1]
                    / (ac_subpop_filt[pop1][:, 0] + ac_subpop_filt[pop1][:, 1])
                    / (ac_subpop_filt[pop1][:, 0] + ac_subpop_filt[pop1][:, 1] - 1)
                )
            else:
                H = np.sum(
                    1.0
                    * ac_subpop_filt[pop1][:, 0]
                    * ac_subpop_filt[pop2][:, 1]
                    / (ac_subpop_filt[pop1][:, 0] + ac_subpop_filt[pop1][:, 1])
                    / (ac_subpop_filt[pop2][:, 0] + ac_subpop_filt[pop2][:, 1])
                    + 1.0
                    * ac_subpop_filt[pop1][:, 1]
                    * ac_subpop_filt[pop2][:, 0]
                    / (ac_subpop_filt[pop1][:, 0] + ac_subpop_filt[pop1][:, 1])
                    / (ac_subpop_filt[pop2][:, 0] + ac_subpop_filt[pop2][:, 1])
                )
            Hs[(pop1, pop2)] = H

    return Hs


def _get_reported_stats(
    genotypes,
    bins,
    sample_ids,
    positions=None,
    pos_rs=None,
    pop_file=None,
    pops=None,
    use_genotypes=True,
    report=True,
    report_spacing=1000,
    use_cache=True,
    stats_to_compute=None,
    ac_filter=True,
):
    ### build wrapping function that can take use_cache = True or False
    # now if bins is empty, we only return heterozygosity statistics

    if stats_to_compute == None:
        if pops is None:
            stats_to_compute = Util.moment_names(1)
        else:
            stats_to_compute = Util.moment_names(len(pops))

    bs = list(zip(bins[:-1], bins[1:]))

    if use_cache == True:
        type_counts = _count_types_sparse(
            genotypes,
            bins,
            sample_ids,
            positions=positions,
            pos_rs=pos_rs,
            pop_file=pop_file,
            pops=pops,
            use_genotypes=use_genotypes,
            report=report,
            report_spacing=report_spacing,
            use_cache=use_cache,
        )

        sums = _get_ld_stat_sums(
            type_counts,
            stats_to_compute[0],
            bins,
            use_genotypes=use_genotypes,
            report=report,
        )

    else:
        sums = _count_types_sparse(
            genotypes,
            bins,
            sample_ids,
            positions=positions,
            pos_rs=pos_rs,
            pop_file=pop_file,
            pops=pops,
            use_genotypes=use_genotypes,
            report=report,
            report_spacing=report_spacing,
            use_cache=use_cache,
            stats_to_compute=stats_to_compute,
        )

    if report is True:
        print(current_time(), "computed sums\ngetting heterozygosity statistics")
        sys.stdout.flush()

    if len(stats_to_compute[1]) == 0:
        Hs = {}
    else:
        Hs = _get_H_statistics(
            genotypes,
            sample_ids,
            pop_file=pop_file,
            pops=pops,
            ac_filter=ac_filter,
            report=report,
        )

    reported_stats = {}
    reported_stats["bins"] = bs
    reported_stats["sums"] = [np.empty(len(stats_to_compute[0])) for b in bs] + [
        np.empty(len(stats_to_compute[1]))
    ]
    for ii, b in enumerate(bs):
        for s in stats_to_compute[0]:
            reported_stats["sums"][ii][stats_to_compute[0].index(s)] = sums[b][s]

    if pops == None:
        pops = ["ALL"]
    for s in stats_to_compute[1]:
        reported_stats["sums"][-1][stats_to_compute[1].index(s)] = Hs[
            (pops[int(s.split("_")[1])], pops[int(s.split("_")[2])])
        ]
    reported_stats["stats"] = stats_to_compute
    reported_stats["pops"] = pops
    return reported_stats


def compute_ld_statistics(
    vcf_file,
    bed_file=None,
    chromosome=None,
    rec_map_file=None,
    map_name=None,
    map_sep=None,
    pop_file=None,
    pops=None,
    cM=True,
    r_bins=None,
    bp_bins=None,
    min_bp=None,
    use_genotypes=True,
    use_h5=True,
    stats_to_compute=None,
    ac_filter=True,
    report=True,
    report_spacing=1000,
    use_cache=True,
):
    """
    Computes LD statistics for a given VCF. Binning can be done by base pair
    or recombination distances, the latter requiring a recombination map. For
    more than one population, we include a population file that maps samples
    to populations, and specify with populations to compute statistics fro.

    If data is phased, we can set ``use_genotypes`` to False, and there are
    other options for masking data.

    .. note::
        Currently, the recombination map is not given in HapMap format.
        Future versions will accept HapMap formatted recombination maps
        and deprecate some of the boutique handling of map options here.

    :param str vcf_file: The input VCF file name.
    :param str bed_file: An optional bed file that specifies regions over which
        to compute LD statistics. If None, computes statistics for all positions
        in VCF.
    :param str chromosome: If None, treats all positions in VCF as coming from same
        chromosome. If multiple chromosomes are reported in the same VCF, we need to
        specify which chromosome to keep variants from.
    :param str rec_map_file: The input recombination map. The format is
        {pos}\t{map (cM)}\t{additional maps}\n
    :param str map_name: If None, takes the first map column, otherwise takes the
        specified map column with the name matching the recombination map file header.
    :param str map_sep: Deprecated! We now read the recombination map, splitting by
        any white space. Previous behaviour: Tells pandas how to parse the recombination map.
    :param str pop_file: A file the specifies the population for each sample in the VCF.
        Each sample is listed on its own line, in the format "{sample}\t{pop}". The
        first line must be "sample\tpop".
    :param list(str) pops: List of populations to compute statistics for.
        If none are given, it treates every sample as coming from the same population.
    :param bool cM: If True, the recombination map is specified in cM. If False,
        the map is given in units of Morgans.
    :param list(float) r_bins: A list of raw recombination rate bin edges.
    :param list(float) bp_bins: If ``r_bins`` are not given, a list of bp bin
        edges (for use when no recombination map is specified).
    :param int, float min_bp: The minimum bp allowed for a segment specified
        by the bed file.
    :param bool use_genotypes: If True, we assume the data in the VCF is unphased.
        Otherwise, we use phased information.
    :param bool use_h5: If True, we use the h5 format.
    :param list stats_to_compute: If given, we compute only the statistics specified.
        Otherwise, we compute all possible statistics for the populations given.
    :param ac_filter: Ensure at least two samples are present per population. This
        prevents computed heterozygosity statistics from returning NaN when some
        loci have too few called samples.
    :param bool report: If True, we report the progress of our parsing.
    :param int report_spacing: We track the number of "left" variants we compute,
        and report our progress with the given spacing.
    :param bool use_cache: If True, cache intermediate results.
    """

    check_imports()

    if r_bins is None and bp_bins is None:
        warnings.warn(
            "Both r_bins and bp_pins are None, so no LD statistics will be computed"
        )
        bins = []
    elif r_bins is not None:
        if bp_bins is not None:
            raise ValueError("Can specify only recombination or bp bins, not both")

    positions, genotypes, counts, sample_ids = get_genotypes(
        vcf_file,
        bed_file=bed_file,
        chromosome=chromosome,
        min_bp=min_bp,
        use_h5=use_h5,
        report=report,
    )

    if report == True:
        print(current_time(), "kept {0} total variants".format(len(positions)))
        sys.stdout.flush()

    if rec_map_file is not None and r_bins is not None:
        if report is True:
            print(current_time(), "assigning recombination rates to positions")
            sys.stdout.flush()
        pos_rs = _assign_recombination_rates(
            positions, rec_map_file, map_name=map_name, cM=cM, report=report
        )
        bins = r_bins
    else:
        if report is True:
            print(
                current_time(), "no recombination map provided, using physical distance"
            )
            sys.stdout.flush()
        pos_rs = None
        if bp_bins is not None:
            bins = bp_bins
        else:
            bins = []

    if not np.all([b1 - b0 > 0 for b0, b1 in zip(bins[:-1], bins[1:])]):
        raise ValueError("bins must be a monotonically increasing list")

    reported_stats = _get_reported_stats(
        genotypes,
        bins,
        sample_ids,
        positions=positions,
        pos_rs=pos_rs,
        pop_file=pop_file,
        pops=pops,
        use_genotypes=use_genotypes,
        report=report,
        stats_to_compute=stats_to_compute,
        report_spacing=report_spacing,
        use_cache=use_cache,
        ac_filter=ac_filter,
    )

    return reported_stats


def means_from_region_data(all_data, stats, norm_idx=0):
    """
    Get means over all parsed regions.

    :param dict all_data: A dictionary with keys as unique identifiers of the
        regions and values as reported stats from ``compute_ld_statistics``.
    :param list of lists stats: The list of LD and H statistics that are present
        in the data replicates.
    :param int, optional norm_idx: The index of the population to normalize by.
    """
    norm_stats = ["pi2_{0}_{0}_{0}_{0}".format(norm_idx), "H_{0}_{0}".format(norm_idx)]
    means = [0 * sums for sums in all_data[list(all_data.keys())[0]]["sums"]]
    for reg in all_data.keys():
        for ii in range(len(means)):
            means[ii] += all_data[reg]["sums"][ii]

    for ii in range(len(means) - 1):
        means[ii] /= means[ii][stats[0].index(norm_stats[0])]
    means[-1] /= means[-1][stats[1].index(norm_stats[1])]
    return means


def get_bootstrap_sets(
    all_data,
    num_bootstraps=None,
    normalization=0,
    remove_norm_stats=True,
    remove_Dz=False,
):
    """
    From a dictionary of all the regional data, resample with replacement
    to construct bootstrap data.

    Returns a list of bootstrapped datasets of mean statistics.

    :param dict all_data: Dictionary of regional LD statistics. Keys are region
        identifiers and must be unique, and the items are the outputs of
        ``compute_ld_statistics``.
    :param int num_bootstraps: The number of bootstrap replicates to compute. If
        None, it computes the same number as the nubmer of regions in ``all_data``.
    :param int normalization: The index of the population to normalize by. Defaults
        to 0.
    :param bool remove_norm_stats: If we should remove the stat used for normalization.
    :param bool remove_Dz: If we should remove Dz statistics.
    """
    regions = list(all_data.keys())
    reg = regions[0]
    stats = all_data[reg]["stats"]

    num_regions = len(all_data)
    if num_bootstraps is None:
        num_bootstraps = num_regions

    all_boot = []

    for rep in range(num_bootstraps):
        temp_data = {}
        choices = np.random.choice(regions, num_regions, replace=True)
        for i, c in enumerate(choices):
            temp_data[i] = all_data[c]
        boot_means = means_from_region_data(temp_data, stats, normalization)

        delete_ld = []
        delete_h = []
        if remove_norm_stats:
            delete_ld.append(
                stats[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))
            )
            delete_h.append(stats[1].index("H_{0}_{0}".format(normalization)))
        if remove_Dz:
            for i, stat in enumerate(stats[0]):
                if stat.split("_")[0] == "Dz":
                    delete_ld.append(i)
        if len(delete_ld) > 0:
            for ii in range(len(boot_means) - 1):
                boot_means[ii] = np.delete(boot_means[ii], delete_ld)
        if len(delete_h) > 0:
            boot_means[-1] = np.delete(boot_means[-1], delete_h)

        all_boot.append(boot_means)

    return all_boot


def bootstrap_data(all_data, normalization=0):
    """
    Returns bootstrapped variances for LD statistics. This function operates
    on data that is sums (i.e. the direct output of ``compute_ld_statistics()``),
    instead of mean statistics.

    We first check that all 'stats', 'bins', 'pops' (if present),
    match across all regions

    If there are N total regions, we compute N bootstrap replicates
    by sampling N times with replacement and summing over all 'sums'.

    :param dict all_data: A dictionary (with arbitrary keys), where each value
        is LD statistics computed from a distinct region. all_data[reg]
        stats from each region has keys, 'bins', 'sums', 'stats', and
        optional 'pops'.
    :param int normalization: we work with :math:`\\sigma_d^2` statistics,
        and by default we use population 0 to normalize stats
    """
    # check that var-cov matrix will be able to be computed
    k = list(all_data.keys())[0]
    for v in all_data[k]["sums"]:
        if len(all_data) < len(v):
            raise ValueError(
                "There are not enough independent regions to compute "
                "variance-covariance matrix"
            )

    norm_stats = [
        "pi2_{0}_{0}_{0}_{0}".format(normalization),
        "H_{0}_{0}".format(normalization),
    ]

    regions = list(all_data.keys())
    reg = regions[0]
    stats = all_data[reg]["stats"]
    N = len(regions)

    means = means_from_region_data(all_data, stats, normalization)

    # construct bootstrap data
    bootstrap_data = [np.zeros((len(sums), N)) for sums in means]

    for boot_num in range(N):
        boot_means = [0 * sums for sums in means]
        samples = np.random.choice(regions, N)
        for reg in samples:
            for ii in range(len(boot_means)):
                boot_means[ii] += all_data[reg]["sums"][ii]

        for ii in range(len(boot_means) - 1):
            boot_means[ii] /= boot_means[ii][stats[0].index(norm_stats[0])]
        boot_means[-1] /= boot_means[-1][stats[1].index(norm_stats[1])]

        for ii in range(len(boot_means)):
            bootstrap_data[ii][:, boot_num] = boot_means[ii]

    varcovs = [np.cov(bootstrap_data[ii]) for ii in range(len(bootstrap_data))]

    mv = {}
    mv["bins"] = all_data[reg]["bins"]
    mv["stats"] = all_data[reg]["stats"]
    if "pops" in all_data[reg]:
        mv["pops"] = all_data[reg]["pops"]
    mv["means"] = means
    mv["varcovs"] = varcovs

    return mv


def subset_data(
    data, pops_to, normalization=0, r_min=None, r_max=None, remove_Dz=False
):
    """
    Take the output data and get r_edges, ms, vcs, and stats to pass to inference
    machinery. ``pops_to`` are the subset of the populations to marginalize the data
    to. ``r_min`` and ``r_max`` trim bins that fall outside of this range, and
    ``remove_Dz`` allows us to remove all :math:`\\sigma_{Dz}` statistics.

    :param data: The output of ``bootstrap_data``, which contains
        bins, statistics, populations, means, and variance-covariance matrices.
    :param pops_to: A list of populations to subset to.
    :param normalization: The population index that the original data was
        normalized by.
    :param r_min: The minimum recombination distance to keep.
    :param r_max: The maximum recombination distance to keep.
    :param remove_Dz: If True, remove all Dz statistics. Otherwise keep them.
    """
    pops_from = data["pops"]
    if not np.all([p in pops_from for p in pops_to]):
        raise ValueError("All pops in pops_to must be in data")

    new_pop_ids = {}
    for pop in pops_to:
        new_pop_ids[pops_from.index(pop)] = pops_to.index(pop)

    stats = data["stats"]

    to_remove = [[], []]
    new_stats = [[], []]

    for j in [0, 1]:
        for i, stat in enumerate(stats[j]):
            if stat in [
                "pi2_{0}_{0}_{0}_{0}".format(normalization),
                "H_{0}_{0}".format(normalization),
            ]:
                to_remove[j].append(i)
            else:
                if remove_Dz == True:
                    if stat.split("_")[0] == "Dz":
                        to_remove[j].append(i)
                        continue
                p_inds = [int(x) for x in stat.split("_")[1:]]
                if len(set(p_inds) - set(new_pop_ids)) == 0:
                    new_stat = "_".join(
                        [stat.split("_")[0]] + [str(new_pop_ids[x]) for x in p_inds]
                    )
                    new_stats[j].append(new_stat)
                else:
                    to_remove[j].append(i)

    means = []
    varcovs = []

    for i, b in enumerate(data["bins"]):
        if r_min is not None:
            if b[0] < r_min:
                continue
        if r_max is not None:
            if b[1] > r_max:
                continue
        means.append(np.delete(data["means"][i], to_remove[0]))
        varcovs.append(
            np.delete(
                np.delete(data["varcovs"][i], to_remove[0], axis=0),
                to_remove[0],
                axis=1,
            )
        )

    means.append(np.delete(data["means"][-1], to_remove[1]))
    varcovs.append(
        np.delete(
            np.delete(data["varcovs"][-1], to_remove[1], axis=0), to_remove[1], axis=1
        )
    )

    r_edges = np.array(sorted(list(set(np.array(data["bins"]).flatten()))))
    if r_min is not None:
        r_edges = r_edges[r_edges >= r_min]
    if r_max is not None:
        r_edges = r_edges[r_edges <= r_max]

    return r_edges, means, varcovs, new_stats
