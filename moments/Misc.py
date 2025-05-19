"""
Miscellaneous utility functions. Including ms simulation.
"""

import bisect, collections, operator, os, sys, time

import numpy as np
import scipy.linalg

from . import Numerics
from . import Spectrum_mod
import functools

# Nucleotide order assumed in Q matrices.
code = "CGTA"

#: Storage for times at which each stream was flushed.
__times_last_flushed = {}


def delayed_flush(stream=sys.stdout, delay=1):
    """
    Flush a stream, ensuring that it is only flushed every 'delay' *minutes*.
    Note that upon the first call to this method, the stream is not flushed.

    stream: The stream to flush. For this to work with simple 'print'
            statements, the stream should be sys.stdout.
    delay: Minimum time *in minutes* between flushes.

    This function is useful to prevent I/O overload on the cluster.
    """
    global __times_last_flushed

    curr_time = time.time()
    # If this is the first time this method has been called with this stream,
    # we need to fill in the times_last_flushed dict. setdefault will do this
    # without overwriting any entry that may be there already.
    if stream not in __times_last_flushed:
        __times_last_flushed[stream] = curr_time
    last_flushed = __times_last_flushed[stream]

    # Note that time.time() returns values in seconds, hence the factor of 60.
    if (curr_time - last_flushed) >= delay * 60:
        stream.flush()
        __times_last_flushed[stream] = curr_time


def ensure_1arg_func(var):
    """
    Ensure that var is actually a one-argument function.

    This is primarily used to convert arguments that are constants into
    trivial functions of time for use in integrations where parameters are
    allowed to change over time.
    """
    if np.isscalar(var):
        # If a constant was passed in, use lambda to make it a nice
        #  simple function.
        var_f = lambda t: var
    else:
        var_f = var
    if not callable(var_f):
        raise ValueError("Argument is not a constant or a function.")
    try:
        var_f(0.0)
    except TypeError:
        raise ValueError("Argument is not a constant or a one-argument " "function.")
    return var_f


def ms_command(theta, ns, core, iter, recomb=0, rsites=None, seeds=None):
    """
    Generate ms command for simulation from core.

    theta: Assumed theta
    ns: Sample sizes
    core: Core of ms command that specifies demography.
    iter: Iterations to run ms
    recomb: Assumed recombination rate
    rsites: Sites for recombination. If None, default is 10*theta.
    seeds: Seeds for random number generator. If None, ms default is used.
           Otherwise, three integers should be passed. Example: (132, 435, 123)
    """
    warnings.warn(
        "Functions relating to `ms` are deprecated in favor of `demes`, will be "
        "removed in version 1.5",
        category=DeprecationWarning,
    )   

    if len(ns) > 1:
        ms_command = (
            "ms %(total_chrom)i %(iter)i -t %(theta)f -I %(numpops)i "
            "%(sample_sizes)s %(core)s"
        )
    else:
        ms_command = "ms %(total_chrom)i %(iter)i -t %(theta)f  %(core)s"

    if recomb:
        ms_command = ms_command + " -r %(recomb)f %(rsites)i"
        if not rsites:
            rsites = theta * 10
    sub_dict = {
        "total_chrom": np.sum(ns),
        "iter": iter,
        "theta": theta,
        "numpops": len(ns),
        "sample_sizes": " ".join(map(str, ns)),
        "core": core,
        "recomb": recomb,
        "rsites": rsites,
    }

    ms_command = ms_command % sub_dict

    if seeds is not None:
        seed_command = " -seeds %i %i %i" % (seeds[0], seeds[1], seeds[2])
        ms_command = ms_command + seed_command

    return ms_command


def perturb_params(params, fold=1, lower_bound=None, upper_bound=None):
    """
    Generate a perturbed set of parameters. Each element of params is randomly
    perturbed `fold` factors of 2 up or down.

    :param fold: Number of factors of 2 to perturb by, defaults to 1.
    :type fold: float, optional
    :param lower_bound: If not None, the resulting parameter set is adjusted
        to have all value greater than lower_bound.
    :type lower_bound: list of floats, optional
    :param upper_bound: If not None, the resulting parameter set is adjusted
        to have all value less than upper_bound.
    :type upper_bound: list of floats, optional
    """
    pnew = params * 2 ** (fold * (2 * np.random.random(len(params)) - 1))
    if lower_bound is not None:
        for ii, bound in enumerate(lower_bound):
            if bound is None:
                lower_bound[ii] = -np.inf
        pnew = np.maximum(pnew, 1.01 * np.asarray(lower_bound))
    if upper_bound is not None:
        for ii, bound in enumerate(upper_bound):
            if bound is None:
                upper_bound[ii] = np.inf
        pnew = np.minimum(pnew, 0.99 * np.asarray(upper_bound))
    return pnew


def make_fux_table(fid, ts, Q, tri_freq):
    """
    Make file of 1-fux for use in ancestral misidentification correction.

    fid: Filename to output to.
    ts: Expected number of substitutions per site between ingroup and outgroup.
    Q: Trinucleotide transition rate matrix. This should be a 64x64 matrix, in
       which entries are ordered using the code CGTA -> 0,1,2,3. For example,
       ACT -> 3*16+0*4+2*1=50. The transition rate from ACT to AGT is then
       entry 50,54.
    tri_freq: Dictionary in which each entry maps a trinucleotide to its
              ancestral frequency. e.g. {'AAA': 0.01, 'AAC':0.012...}
              Note that should be the frequency in the entire region scanned
              for variation, not just sites where there are SNPs.
    """
    warnings.warn(
        "Operations using the `data_dict` are deprecated and will be removed "
        "in version 1.5, in favor of `from_vcf` and associated functions "
        "in the `Parsing` module",
        category=DeprecationWarning,
    )   

    # Ensure that the *columns* of Q sum to zero.
    # That is the correct condition when Q_{i,j} is the rate from i to j.
    # This indicates a typo in Hernandez, Williamson, and Bustamante.
    for ii in range(Q.shape[1]):
        s = Q[:, ii].sum() - Q[ii, ii]
        Q[ii, ii] = -s

    eQhalf = scipy.linalg.matfuncs.expm(Q * ts / 2.0)
    if not hasattr(fid, "write"):
        newfile = True
        fid = open(fid, "w")

    outlines = []
    for first_ii, first in enumerate(code):
        for x_ii, x in enumerate(code):
            for third_ii, third in enumerate(code):
                # This the index into Q and eQ
                xind = 16 * first_ii + 4 * x_ii + 1 * third_ii
                for u_ii, u in enumerate(code):
                    # This the index into Q and eQ
                    uind = 16 * first_ii + 4 * u_ii + 1 * third_ii

                    ## Note that the Q terms factor out in our final
                    ## calculation, because for both PMuUu and PMuUx the final
                    ## factor in Eqn 2 is P(S={u,x}|M=u).
                    # Qux = Q[uind,xind]
                    # denomu = Q[uind].sum() - Q[uind,uind]

                    PMuUu, PMuUx = 0, 0
                    # Equation 2 in HWB. We have to generalize slightly to
                    # calculate PMuUx. In calculate PMuUx, we're summing over
                    # alpha the probability that the MRCA was alpha, and it
                    # substituted to x on the outgroup branch, and it
                    # substituted to u on the ingroup branch, and it mutated to
                    # x in the ingroup (conditional on it having mutated in the
                    # ingroup). Note that the mutation to x condition cancels
                    # in fux, so we don't bother to calculate it.
                    for aa, alpha in enumerate(code):
                        aind = 16 * first_ii + 4 * aa + 1 * third_ii

                        pia = tri_freq[first + alpha + third]
                        Pau = eQhalf[aind, uind]
                        Pax = eQhalf[aind, xind]

                        PMuUu += pia * Pau * Pau
                        PMuUx += pia * Pau * Pax

                    # This is 1-fux. For a given SNP with actual ancestral state
                    # u and derived allele x, this is 1 minus the probability
                    # that the outgroup will have u.
                    # Eqn 3 in HWB.
                    res = 1 - PMuUu / (PMuUu + PMuUx)
                    # These aren't SNPs, so we can arbitrarily set them to 0
                    if u == x:
                        res = 0

                    outlines.append("%c%c%c %c %.6f" % (first, x, third, u, res))

    fid.write(os.linesep.join(outlines))
    if newfile:
        fid.close()


def zero_diag(Q):
    """
    Copy of Q altered such that diagonal entries are all 0.
    """
    Q_nodiag = Q.copy()
    for ii in range(Q.shape[0]):
        Q_nodiag[ii, ii] = 0
    return Q_nodiag


def tri_freq_dict_to_array(tri_freq_dict):
    """
    Convert dictionary of trinucleotide frequencies to array in correct order.
    """
    tripi = np.zeros(64)
    for ii, left in enumerate(code):
        for jj, center in enumerate(code):
            for kk, right in enumerate(code):
                row = ii * 16 + jj * 4 + kk
                tripi[row] = tri_freq_dict[left + center + right]
    return tripi


def total_instantaneous_rate(Q, pi):
    """
    Total instantaneous substitution rate.
    """
    Qzero = zero_diag(Q)
    return np.dot(pi, Qzero).sum()


def make_data_dict(filename):
    """
    Parse a file containing genomic sequence information in the format described
    by the wiki, and store the information in a properly formatted dictionary.

    filename: Name of file to work with.

    The file can be zipped (extension .zip) or gzipped (extension .gz). If
    zipped, there must be only a single file in the zip archive.
    """
    warnings.warn(
        "Operations using the `data_dict` are deprecated and will be removed "
        "in version 1.5, in favor of `from_vcf` and associated functions "
        "in the `Parsing` module",
        category=DeprecationWarning,
    )   

    if os.path.splitext(filename)[1] == ".gz":
        import gzip

        f = gzip.open(filename)
    elif os.path.splitext(filename)[1] == ".zip":
        import zipfile

        archive = zipfile.ZipFile(filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError(
                "Must be only a single data file in zip " "archive: %s" % filename
            )
        f = archive.open(namelist[0])
    else:
        f = open(filename)

    # Skip to the header
    while True:
        header = f.readline()
        if not header.startswith("#"):
            break

    allele2_index = header.split().index("Allele2")

    # Pull out our pop ids
    pops = header.split()[3:allele2_index]

    # The empty data dictionary
    data_dict = {}

    # Now walk down the file
    for SNP_ii, line in enumerate(f):
        if line.startswith("#"):
            continue
        # Split the into fields by whitespace
        spl = line.split()

        data_this_snp = {}

        # We convert to upper case to avoid any issues with mixed case between
        # SNPs.
        data_this_snp["context"] = spl[0].upper()
        data_this_snp["outgroup_context"] = spl[1].upper()
        data_this_snp["outgroup_allele"] = spl[1][1].upper()
        data_this_snp["segregating"] = spl[2].upper(), spl[allele2_index].upper()

        calls_dict = {}
        for ii, pop in enumerate(pops):
            calls_dict[pop] = int(spl[3 + ii]), int(spl[allele2_index + 1 + ii])
        data_this_snp["calls"] = calls_dict

        # We name our SNPs using the final columns
        snp_key = (
            spl[allele2_index + len(pops) + 1],
            spl[allele2_index + len(pops) + 2],
        )
        if snp_key == "":
            snp_key = ("SNP", f"{SNP_ii}")

        data_dict[snp_key] = data_this_snp

    return data_dict


def count_data_dict(data_dict, pop_ids):
    """
    Summarize data in data_dict by mapping SNP configurations to counts.

    Returns a dictionary with keys (successful_calls, derived_calls,
    polarized) mapping to counts of SNPs. Here successful_calls is a tuple
    with the number of good calls per population, derived_calls is a tuple
    of derived calls per pop, and polarized indicates whether that SNP was
    polarized using an ancestral state.

    :param data_dict: data_dict formatted as in Misc.make_data_dict
    :type data_dict: data dictionary
    :param pop_ids: IDs of populations to collect data for
    :type pop_ids: list of strings
    """
    warnings.warn(
        "Operations using the `data_dict` are deprecated and will be removed "
        "in version 1.5, in favor of `from_vcf` and associated functions "
        "in the `Parsing` module",
        category=DeprecationWarning,
    )   

    count_dict = collections.defaultdict(int)
    for snp_key, snp_info in data_dict.items():
        # Skip SNPs that aren't biallelic.
        if len(snp_info["segregating"]) != 2:
            continue

        allele1, allele2 = snp_info["segregating"]
        if (
            "outgroup_allele" in snp_info
            and snp_info["outgroup_allele"] != "-"
            and snp_info["outgroup_allele"] in snp_info["segregating"]
        ):
            outgroup_allele = snp_info["outgroup_allele"]
            this_snp_polarized = True
        else:
            outgroup_allele = allele1
            this_snp_polarized = False

        # Extract the allele calls for each population.
        allele1_calls = [snp_info["calls"][pop][0] for pop in pop_ids]
        allele2_calls = [snp_info["calls"][pop][1] for pop in pop_ids]
        # How many chromosomes did we call successfully in each population?
        successful_calls = [a1 + a2 for (a1, a2) in zip(allele1_calls, allele2_calls)]

        # Which allele is derived (different from outgroup)?
        if allele1 == outgroup_allele:
            derived_calls = allele2_calls
        elif allele2 == outgroup_allele:
            derived_calls = allele1_calls

        # Update count_dict
        count_dict[
            tuple(successful_calls), tuple(derived_calls), this_snp_polarized
        ] += 1
    return count_dict


def make_data_dict_vcf(
    vcf_filename,
    popinfo_filename,
    filter=True,
    flanking_info=[None, None],
    skip_multiallelic=True,
):
    """
    Parse a VCF file containing genomic sequence information, along with a file
    identifying the population of each sample, and store the information in
    a properly formatted dictionary.

    Each file may be zipped (.zip) or gzipped (.gz). If a file is zipped,
    it must be the only file in the archive, and the two files cannot be zipped
    together. Both files must be present for the function to work.

    :param vcf_filename: Name of VCF file to work with. The function currently works
        for biallelic SNPs only, so if REF or ALT is anything other
        than a single base pair (A, C, T, or G), the allele will be
        skipped. Additionally, genotype information must be present
        in the FORMAT field GT, and genotype info must be known for
        every sample, else the SNP will be skipped. If the ancestral
        allele is known it should be specified in INFO field 'AA'.
        Otherwise, it will be set to '-'.
    :type vcf_filename: str
    :param popinfo_filename: Name of file containing the population assignments for
        each sample in the VCF. If a sample in the VCF file does
        not have a corresponding entry in this file, it will be
        skipped. See _get_popinfo for information on how this
        file must be formatted.
    :type popinfo_filename: str
    :param filter: If set to True, alleles will be skipped if they have not passed
        all filters (i.e. either 'PASS' or '.' must be present in FILTER column.
    :type filter: bool, optional
    :param flanking_info: Flanking information for the reference and/or ancestral
        allele can be provided as field(s) in the INFO column. To
        add this information to the dict, flanking_info should
        specify the names of the fields that contain this info as a
        list (e.g. ['RFL', 'AFL'].) If context info is given for
        only one allele, set the other item in the list to None,
        (e.g. ['RFL', None]). Information can be provided as a 3
        base-pair sequence or 2 base-pair sequence, where the first
        base-pair is the one immediately preceding the SNP, and the
        last base-pair is the one immediately following the SNP.
    :type flanking_info: list of strings, optional
    :param skip_multiallelic: If True, only keep biallelic sites, and skip sites that
        have more than one ALT allele.
    :type skip_multiallelic: bool, optional
    """
    warnings.warn(
        "Operations using the `data_dict` are deprecated and will be removed "
        "in version 1.5, in favor of `from_vcf` and associated functions "
        "in the `Parsing` module",
        category=DeprecationWarning,
    )   

    if not skip_multiallelic:
        raise ValueError(
            "We can only keep biallelic sites, and multiallelic tallying is not "
            "currently supported. Set skip_multiallelic to True."
        )

    # Read population information from file based on extension
    if os.path.splitext(popinfo_filename)[1] == ".gz":
        import gzip

        popinfo_file = gzip.open(popinfo_filename)
    elif os.path.splitext(popinfo_filename)[1] == ".zip":
        import zipfile

        archive = zipfile.ZipFile(popinfo_filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError(
                "Must be only a single popinfo file in zip "
                "archive: {}".format(popinfo_filename)
            )
        popinfo_file = archive.open(namelist[0])
    else:
        popinfo_file = open(popinfo_filename)
    # pop_dict has key, value pairs of "SAMPLE_NAME" : "POP_NAME"
    popinfo_dict = _get_popinfo(popinfo_file)
    popinfo_file.close()

    # Open VCF file
    if os.path.splitext(vcf_filename)[1] == ".gz":
        import gzip

        vcf_file = gzip.open(vcf_filename)
    elif os.path.splitext(vcf_filename)[1] == ".zip":
        import zipfile

        archive = zipfile.ZipFile(vcf_filename)
        namelist = archive.namelist()
        if len(namelist) != 1:
            raise ValueError(
                "Must be only a single vcf file in zip "
                "archive: {}".format(vcf_filename)
            )
        vcf_file = archive.open(namelist[0])
    else:
        vcf_file = open(vcf_filename)

    data_dict = {}
    for line in vcf_file:
        # decoding lines for Python 3 - probably a better way to handle this
        try:
            line = line.decode()
        except AttributeError:
            pass
        # Skip metainformation
        if line.startswith("##"):
            continue
        # Read header
        if line.startswith("#"):
            header_cols = line.split()
            # Ensure there is at least one sample
            if len(header_cols) <= 9:
                raise ValueError("No samples in VCF file")
            # Use popinfo_dict to get the order of populations present in VCF
            poplist = [
                popinfo_dict[sample] if sample in popinfo_dict else None
                for sample in header_cols[9:]
            ]
            continue

        # Read SNP data
        cols = line.split()
        snp_key = (cols[0], cols[1])  # (CHROM, POS)
        snp_dict = {}

        # Skip SNP if filter is set to True and it fails a filter test
        if filter and cols[6] != "PASS" and cols[6] != ".":
            continue

        # Add reference and alternate allele info to dict
        ref, alt = (allele.upper() for allele in cols[3:5])
        if ref not in ["A", "C", "G", "T"] or alt not in ["A", "C", "G", "T"]:
            # Skip line if site is not an SNP
            continue
        snp_dict["segregating"] = (ref, alt)
        snp_dict["context"] = "-" + ref + "-"

        # Add ancestral allele information if available
        info = cols[7].split(";")
        for field in info:
            if field.startswith("AA"):
                outgroup_allele = field[3:].upper()
                if outgroup_allele not in ["A", "C", "G", "T"]:
                    # Skip if ancestral not single base A, C, G, or T
                    outgroup_allele = "-"
                break
        else:
            outgroup_allele = "-"
        snp_dict["outgroup_allele"] = outgroup_allele
        snp_dict["outgroup_context"] = "-" + outgroup_allele + "-"

        # Add flanking info if it is present
        rflank, aflank = flanking_info
        for field in info:
            if rflank and field.startswith(rflank):
                flank = field[len(rflank + 1) :].upper()
                if not (len(flank) == 2 or len(flank) == 3):
                    continue
                prevb, nextb = flank[0], flank[-1]
                if prevb not in ["A", "C", "T", "G"]:
                    prevb = "-"
                if nextb not in ["A", "C", "T", "G"]:
                    nextb = "-"
                snp_dict["context"] = prevb + ref + nextb
                continue
            if aflank and field.startswith(aflank):
                flank = field[len(aflank + 1) :].upper()
                if not (len(flank) == 2 or len(flank) == 3):
                    continue
                prevb, nextb = flank[0], flank[-1]
                if prevb not in ["A", "C", "T", "G"]:
                    prevb = "-"
                if nextb not in ["A", "C", "T", "G"]:
                    nextb = "-"
                snp_dict["outgroup_context"] = prevb + outgroup_allele + nextb

        # Add reference and alternate allele calls for each population
        calls_dict = {}
        gtindex = cols[8].split(":").index("GT")
        for pop, sample in zip(poplist, cols[9:]):
            if pop is None:
                continue
            gt = sample.split(":")[gtindex]
            g1, g2 = gt[0], gt[2]
            if pop not in calls_dict:
                calls_dict[pop] = (0, 0)
            refcalls, altcalls = calls_dict[pop]
            refcalls += int(g1 == "0") + int(g2 == "0")
            altcalls += int(g1 == "1") + int(g2 == "1")
            calls_dict[pop] = (refcalls, altcalls)
        snp_dict["calls"] = calls_dict
        data_dict[snp_key] = snp_dict

    vcf_file.close()
    return data_dict


def _get_popinfo(popinfo_file):
    """
    Helper function for make_data_dict_vcf. Takes an open file that contains
    information on the population designations of each sample within a VCF file,
    and returns a dictionary containing {"SAMPLE_NAME" : "POP_NAME"} pairs.

    The file should be formatted as a table, with columns delimited by
    whitespace, and rows delimited by new lines. Lines beginning with '#' are
    considered comments and will be ignored. Each sample must appear on its own
    line. If no header information is provided, the first column will be assumed
    to be the SAMPLE_NAME column, while the second column will be assumed to be
    the POP_NAME column. If a header is present, it must be the first
    non-comment line of the file. The column positions of the words "SAMPLE" and
    "POP" (ignoring case) in this header will be used to determine proper
    positions of the SAMPLE_NAME and POP_NAME columns in the table.

    popinfo_file : An open text file of the format described above.
    """
    warnings.warn(
        "Operations using the `data_dict` are deprecated and will be removed "
        "in version 1.5, in favor of `from_vcf` and associated functions "
        "in the `Parsing` module",
        category=DeprecationWarning,
    )
    
    popinfo_dict = {}
    sample_col = 0
    pop_col = 1
    header = False

    # check for header info
    for line in popinfo_file:
        if line.startswith("#"):
            continue
        cols = [col.lower() for col in line.split()]
        if "sample" in cols:
            header = True
            sample_col = cols.index("sample")
        if "pop" in cols:
            header = True
            pop_col = cols.index("pop")
        break

    # read in population information for each sample
    popinfo_file.seek(0)
    for line in popinfo_file:
        if line.startswith("#"):
            continue
        cols = line.split()
        sample = cols[sample_col]
        pop = cols[pop_col]
        # avoid adding header to dict
        if (sample.lower() == "sample" or pop.lower() == "pop") and header:
            header = False
            continue
        popinfo_dict[sample] = pop

    return popinfo_dict


def bootstrap(
    data_dict,
    pop_ids,
    projections,
    mask_corners=True,
    polarized=True,
    bed_filename=None,
    num_boots=100,
    save_dir=None,
):
    """
    Use a non-parametric bootstrap on SNP information contained in a dictionary
    to generate new data sets. The new data is created by sampling with
    replacement from independent units of the original data. These units can
    simply be chromosomes, or they can be regions specified in a BED file.

    This function either returns a list of all the newly created SFS, or writes
    them to disk in a specified directory.

    See :func:`moments.Spectrum.from_data_dict` for more details about the options for
    creating spectra.

    :param data_dict: Dictionary containing properly formatted SNP information (i.e.
        created using one of the make_data_dict methods).
    :type data_dict: dict of SNP information
    :param pop_ids: List of population IDs.
    :type pop_ids: list of strings
    :param projections: Projection sizes for the given population IDs.
    :type projections: list of ints
    :param mask_corners: If True, mask the invariant bins of the SFS.
    :type mask_corners: bool, optional
    :param polarized: If True, we assume we know the ancestral allele. If False,
        return folded spectra.
    :type polarized: bool, optional
    :param bed_filename: If None, chromosomes will be used as the units for
        resampling. Otherwise, this should be the filename of a BED
        file specifying the regions to be used as resampling units.
        Chromosome names must be consistent between the BED file and
        the data dictionary, or bootstrap will not work. For example,
        if an entry in the data dict has ID X_Y, then the value in
        in the chromosome field of the BED file must also be X (not
        chrX, chromosomeX, etc.).
        If the name field is provided in the BED file, then any
        regions with the same name will be considered to be part of
        the same unit. This may be useful for sampling as one unit a
        gene that is located across non-continuous regions.
    :type bed_filename: string as path to bed file
    :param num_boots: Number of resampled SFS to generate.
    :type num_boots: int, optional
    :param save_dir: If None, the SFS are returned as a list. Otherwise this should be
        a string specifying the name of a new directory under which all
        of the new SFS should be saved.
    :type save_dir: str, optional
    """
    warnings.warn(
        "Operations using the `data_dict` are deprecated and will be removed "
        "in version 1.5, in favor of `from_vcf` and associated functions "
        "in the `Parsing` module",
        category=DeprecationWarning,
    )   

    # Read in information from BED file if present and store by chromosome
    if bed_filename is not None:
        bed_file = open(bed_filename)
        bed_info_dict = {}
        for linenum, line in enumerate(bed_file):
            fields = line.split()
            # Read in mandatory fields
            chrom, start, end = fields[:3]
            start = int(start)
            end = int(end)
            # Read label info if present, else assign unique label by line number
            label = linenum
            if len(fields) >= 4:
                label = fields[3]
            # Add information to the appropriate chromosome
            if chrom not in bed_info_dict:
                bed_info_dict[chrom] = []
            bed_info_dict[chrom].append((start, end, label))
        bed_file.close()

        # Sort entries by start position, for easier location of proper region
        start_dict = {}
        for chrom, bed_info in bed_info_dict.items():
            bed_info.sort(key=lambda k: k[0])
            start_dict[chrom] = [region[0] for region in bed_info]
        # Dictionary will map region labels to the SNPs contained in that region
        region_dict = {}
        # Iterate through data_dict and add SNPs to proper region
        for snp_key in data_dict:
            chrom, pos = snp_key
            pos = int(pos)
            # Quickly locate proper region in sorted list
            loc = bisect.bisect_right(start_dict[chrom], pos) - 1
            if loc >= 0 and bed_info_dict[chrom][loc][1] >= pos:
                label = bed_info_dict[chrom][loc][2]
                if label not in region_dict:
                    region_dict[label] = []
                region_dict[label].append(snp_key)
    # Separate by chromosome if no BED file provided
    else:
        region_dict = {}
        for snp_key in data_dict:
            chrom, pos = snp_key
            if chrom not in region_dict:
                region_dict[chrom] = []
            region_dict[chrom].append(snp_key)

    # Each entry of list represents single region, with a tuple
    # containing the IDs of all SNPs in the region.
    sample_regions = [tuple(val) for key, val in region_dict.items()]
    num_regions = len(sample_regions)
    if save_dir is None:
        new_sfs_list = []
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Repeatedly resample regions to create new data sets
    for bootnum in range(num_boots):
        # Set up new SFS
        npops = len(pop_ids)
        new_sfs = np.zeros(np.asarray(projections) + 1)
        # Make random selection of regions (with replacement)
        choices = np.random.randint(0, num_regions, num_regions)
        # For each selected region, add its SNP info to SFS
        for choice in choices:
            for snp_key in sample_regions[choice]:
                snp_info = data_dict[snp_key]
                # Skip SNPs that aren't biallelic.
                if len(snp_info["segregating"]) != 2:
                    continue

                allele1, allele2 = snp_info["segregating"]
                if not polarized:
                    # If not polarizing, derived allele is arbitrary
                    outgroup_allele = allele1
                elif (
                    "outgroup_allele" in snp_info
                    and snp_info["outgroup_allele"] != "-"
                    and snp_info["outgroup_allele"] in snp_info["segregating"]
                ):
                    # Otherwise check that it is a useful outgroup
                    outgroup_allele = snp_info["outgroup_allele"]
                else:
                    # If polarized and without good outgroup, skip SNP
                    continue

                # Extract allele calls for each population.
                allele1_calls = [snp_info["calls"][pop][0] for pop in pop_ids]
                allele2_calls = [snp_info["calls"][pop][1] for pop in pop_ids]
                successful_calls = [
                    a1 + a2 for (a1, a2) in zip(allele1_calls, allele2_calls)
                ]
                derived_calls = (
                    allele2_calls if allele1 == outgroup_allele else allele1_calls
                )

                # Slicing allows handling of arbitray population numbers
                slices = [[np.newaxis] * npops for i in range(npops)]
                for i in range(npops):
                    slices[i][i] = slice(None, None, None)

                # Do projections for this SNP
                pop_contribs = []
                call_iter = zip(projections, successful_calls, derived_calls)
                for pop_index, (p_to, p_from, hits) in enumerate(call_iter):
                    contrib = Numerics._cached_projection(p_to, p_from, hits)[
                        tuple(slices[pop_index])
                    ]
                    pop_contribs.append(contrib)
                new_sfs += functools.reduce(operator.mul, pop_contribs)

        new_sfs = Spectrum_mod.Spectrum(
            new_sfs, mask_corners=mask_corners, pop_ids=pop_ids
        )
        if not polarized:
            new_sfs.fold()
        if save_dir is None:
            new_sfs_list.append(new_sfs)
        else:
            filename = "{}/SFS_{}".format(save_dir, bootnum)
            new_sfs.to_file(filename)

    return new_sfs_list if save_dir is None else None


def flip_ancestral_misid(fs, p_misid):
    if p_misid < 0 or p_misid > 1:
        raise ValueError(
            "probability of misidentification must be between zero and one."
        )
    fs_misid = (1 - p_misid) * fs + p_misid * np.flip(fs)
    return fs_misid
