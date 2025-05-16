"""
Contains functions for computing the SFS and sequence length ``L`` from data.

`parse_vcf` calls the function `_tally_vcf` to build a nested dictionary 
representation of the counts of derived alleles observed in a VCF file, then
constucts an SFS of appropriate dimension from this object with the function
`_spectrum_from_tally`. Users can provide arbitrary configurations of 
populations, filter by quality/annotations, restrict sites using a BED file 
and/or a half-open interval, and provide an estimated ancestral sequence in 
several ways detailed in the docstrings. When calling `_spectrum_from_tally`, 
users may also include sites with some missing or filtered data in the output
SFS by specifying a configuration of minimum population-specific sample sizes.
All sites with greater than or equal to this number of samples in each 
population will be included in the SFS. Sites with more samples than specified
have their entries projected down to match the specified size configuration.

`compute_L` calculates the number of callable sites (the effective sequence 
length) ``L`` given a BED file and optionally a half-open interval and the 
coverage of an ancestral sequence in FASTA format. This quantity is useful when
computing the expected SFS using a physical mutation rate. 
"""

from collections import defaultdict
import copy
from datetime import datetime
import gzip
import numpy as np
import os
import sys
import time
import warnings

from . import Spectrum_mod


# Raise each unique warning only once
warnings.simplefilter('once')


def parse_vcf(
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
    ploidy=2,
    verbose=0,
    sample_sizes=None,
    mask_corners=True,
    folded=False
):
    """
    Compute the SFS from genotype data stored in a VCF file. There are 
    several optional parameters for controlling the assignment of ancestral
    states, filtering by quality or annotation, restricting SFS parsing to
    certain intervals, and adjusting the shape of the output SFS.

    There are several ways to specify estimated ancestral states. By 
    default, VCF ``REF`` alleles are interpreted as ancestral alleles. In 
    this case the output SFS should be folded, because the reference allele 
    does not in general correspond to the ancestral allele. If `use_AA` is 
    True, then the ``INFO/AA`` VCF field specifies the ancestral allele. 
    Sites where ``INFO/AA`` is absent or missing data (represented by '.')
    are skipped, raising a warning at the first occurence. If `anc_seq_file` 
    (FASTA format) is given, then ancestral alleles are read from it. Here 
    also, sites are skipped when they lack a valid ancestral allele. This 
    behavior is modulated by `allow_low_confidence`: when True, sites 
    assigned ancestral states represented in lower-case (e.g. 'a') are 
    retained. This usually denotes a low-confidence assignment. Otherwise 
    such sites are skipped. If there are sites in the VCF that fall beyond
    the end of the FASTA sequence which are not otherwise masked or excluded 
    by the `interval` argument, an error is raised.

    When `allow_multiallelic` is False and the ancestral allele is not 
    represented as either the reference or alternate allele at a site, that 
    site is skipped automatically. The relationship between derived alleles
    at such sites is not generally clear.

    The parameter `filters` allows filtering by quality and/or annotation
    at the site and sample level. It should be a flat dictionary with str 
    keys. Key-value pairs may have the following forms:
    'QUAL' should map to a single number (float or int), imposing a minimum 
        value for site ``QUAL`` fields.
    'FILTER' should map to a string or a set, tuple or list of strings. For
        a site to pass, its ``FILTER`` field must equal either the value of 
        'FILTER' (if a string) or that of one of its elements (if a set,
        tuple or list).
    'INFO/FIELD' e.g. 'INFO/GQ' may map to a number, a string, or a set, 
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
    'SAMPLE/FIELD' e.g. 'SAMPLE/GQ' imposes filters at the sample level. 
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
        to, and mutually exclusive with, `pop_file`.
    :type pop_mapping: dict, optional
    :param pop_file: Pathname of a whitespace-separated file mapping samples
        to populations with the format SAMPLE POPULATION. Sample names must
        be unique and there should be one of them on each line (default None
        combines all samples into a single population 'ALL'). Samples 
        present in the VCF but not included here are ignored.
    :type pop_file: str, optional
    :param pops: A list of populations from `pop_file `to parse (default 
        None). Only functions when `pop_file` is given. Populations not in 
        `pops` are ignored. If None, then all populations in `pop_file` are
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
    :param allow_low_confidence: If True (default False) and `anc_seq_file` 
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
        Any VCF sites with sample sizes greater than `sample_sizes` will be 
        projected down to match it. This may be useful when some genotype 
        data is missing or filtered- sites with missing data are otherwise 
        not included in the output SFS. When not given, output sample sizes 
        default to the sample sizes implied by `ploidy` and the number of 
        individuals in each population.
    :type sample_sizes: dict, optional
    :param mask_corners: If True (default), the 'observed in none' and
        'observed in all' entries of the SFS array are masked.
    :type mask_corners: bool, optional
    :param ploidy: Optionally defines the ploidy of samples (default 2).
        Used to determine the haploid sample size from the number of sampled 
        individuals when `sample_sizes` is not given.
    :type plody: int, optional
    :param verbose: If > 0, print a progress message every `verbose` lines
        (default 0).
    :type verbose: int, optional
    :param folded: If True, return the folded SFS (default False).
    :type folded: bool, optional

    :returns: The SFS, represented as a ``moments.Spectrum`` instance.
    :rtype: moments.Spectrum
    """
    data = _tally_vcf(
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
        verbose=verbose,
        ploidy=ploidy
    )
    fs = _spectrum_from_tally(
        data, 
        sample_sizes=sample_sizes, 
        mask_corners=mask_corners
    )
    if folded:
        fs = fs.fold()

    return fs


def compute_L(
    bed_file,
    interval=None,
    anc_seq_file=None,
    allow_low_confidence=False
):
    """
    Compute the sequence length `L` from a BED file.

    If `interval` is given, then only sites within the interval are counted.
    If `anc_seq_file` is given, then only sites which are assigned a valid 
    ancestral state in that file are counted. Whether or not low-confidence 
    assignments- conventionally represented with lower-case letters- are counted 
    is modulated by `allow_low_confidence`.

    :param bed_file: Pathname of a BED file specifying regions over which `L` 
        is to be computed.
    :param interval: 2-list or tuple specifying the interval over which to 
        compute `L` (default None). The lower bound is inclusive, the upper 
        noninclusive, and both are 1-indexed.
    :type interval: list, optional
    :param anc_seq_file: Pathname of a FASTA file holding ancestral nucleotide 
        states (default None). If given, only sites with assigned ancestral 
        states are counted in `L`. Low-confidence assignments are excluded if 
        `allow_low_confidence` is False.
    :type anc_seq_file: str, optional
    :param allow_low_confidence: If True (default False), include nucleotides 
        coded with lower- case letters in the calculation of L. Otherwise, 
        leave them out.
    :type allow_low_confidence: bool, optional

    :returns: Sequence length.
    :rtype: int
    """
    regions, _ = _load_bed_file(bed_file)
    mask = _bed_regions_to_mask(regions)
    if anc_seq_file is not None:
        anc_seq = _load_fasta_file(anc_seq_file)
        if allow_low_confidence:
            seq_mask = ~np.array(
                [nt.capitalize() in ('A', 'T', 'C', 'G') for nt in anc_seq]
            )
        else:
            seq_mask = ~np.array(
                [nt in ('A', 'T', 'C', 'G') for nt in anc_seq]
            )
        length = min(len(mask), len(seq_mask))
        mask = np.maximum(mask[:length], seq_mask[:length])
    if interval is not None:
        if len(interval) != 2:
            raise ValueError('Argument `interval` must have length 2')
        start, end = interval
        if start < 1:
            raise ValueError('Interval must have start > 0')
        # Convert `interval` to 0-indexing
        start0 = start - 1
        end0 = end - 1
        mask = mask[start0:end0]
    L = len(mask) - np.count_nonzero(mask)

    return L


stats = defaultdict(int)


def _clear_stats():
    """
    Re-initialize `stats`.
    """
    global stats
    stats = defaultdict(int)
    return


def _current_time():
    """
    Return a string representing the time and date with yy-mm-dd format.
    """
    return '[' + datetime.strftime(datetime.now(), '%y-%m-%d %H:%M:%S') + ']'


def _print_err(*args, **kwargs):
    """
    Print a (soft) error message to stderr.
    """
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()
    return


def _print_out(*args, **kwargs):
    """
    Print a status report to stdout.
    """
    print(*args, file=sys.stdout, **kwargs)
    sys.stdout.flush()
    return


def _tally_vcf(
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
    ploidy=2,
    verbose=0,
    return_stats=False
):
    """
    Build a dictionary representation of derived allele counts from a VCF file.
    See the documentation of `parse_vcf` for a more detailed discussion of
    polarization, filtering and the treatment of multiallelic sites. The output
    of this function can be transformed into an SFS array using the function
    `spectrum_from_tally`.
    
    Returns a dictionary with keys 'pop_ids', 'sample_sizes' and 'tally'. The 
    `tally` is a nested dictionary which maps configurations of population-
    specific haploid sample sizes to subdictionaries, which in turn map 
    configurations of population-specific derived allele counts to the number of 
    times they are observed. A VCF file may have many different sample sizes if 
    some genotype data are missing, or if filtering is imposed at the sample 
    level. When an ancestral sequence is not provided, counts are of alternate
    rather than derived alleles, and any SFS created from output should be 
    folded to reflect the lack of polarization, as reference alleles do not
    generally correspond to ancestral alleles. Whether alleles are polarized is
    not explicitly tracked by output.

    :param vcf_file: Pathname of the VCF file from which to read. 
    :type vcf_file: str
    :param pop_mapping: Optional dictionary (default None) mapping population 
        IDs to lists of VCF sample IDs. Equivalent in function to, and mutually 
        exclusive with, `pop_file`.
    :type pop_mapping: str, optional
    :param pop_file: Pathname of a whitespace-separated file mapping samples
        to populations with the format SAMPLE POPULATION. Sample names must be
        unique and there should be one of them on each line (default None 
        combines all samples into a single population 'ALL'). Samples present in 
        the VCF but not included here are ignored.
    :type pop_file: str, optional
    :param pops: A list of populations from `pop_file `to parse (default None).
        Only functions when `pop_file` is given. Populations not in `pops` are 
        ignored. If None, then all populations in `pop_file` are included.
    :type pops: list of str
    :param bed_file: Pathname of a BED file defining the intervals within which 
        to parse; useful for applying masks to exclude difficult-to-call or 
        functionally constrained genomic regions (default None). BED files 
        represent intervals as 0-indexed and half-open (the ends of intervals 
        are noninclusive).
    :type bed_file: str, optional
    :param interval: 2-tuple or 2-list specifying a 1-indexed, half-open
        (upper boundary noninclusive) genomic window to parse (default None). 
        May be used in conjuction with a BED file.
    :type interval: list or tuple, optional
    :param anc_seq_file: Pathname of a FASTA file assigning ancestral states
        to sites (default None).
    :type anc_seq_file: str, optional
    :param allow_low_confidence: If True (default False) and `anc_seq_file` is
        given, allows low-confidence ancestral state assignments, represented 
        by lower-case nucleotide codes. If False, these sites are skipped.
    :type allow_low_confidence: bool, optional
    :param use_AA: If True, use the field ``INFO/AA`` specified in the VCF file
        to asign ancestral states (default False). Sites where this field is
        absent or missing are skipped.
    :type use_AA: bool, optional
    :param filters: A dictionary specifying filters on VCF lines or samples;
        its specification is described under `parse_vcf` (default None).
    :type filters: dict, optional
    :param allow_multiallelic: If True (default False), include sites with more
        than one alternate allele as separate entries in the tally. Also allows
        sites where neither the reference nor any alternate allelle(s) matches
        the assigned ancestral state, which are skipped when False.
    :type allow_multiallelic: bool, optional
    :param ploidy: Optional (default 2), defines the maximum derived allele 
        count in combination with the number of sampled individuals.
    :type plody: int, optional
    :param verbose: If > 0, print a progress message every `verbose` lines
        (default 0).
    :type verbose: int, optional
    :param return_stats: If True (default False), return a dictionary of 
        statistics describing the number of sites that were filtered out for 
        various reasons along with `tally`.
    :type return_stats: bool, optional
     
    :return: A dictionary holding population IDs, sample sizes and derived 
        allele counts in the manner described above.
    :rtype: dict
    """
    _clear_stats()

    # Inspect `filters`
    filters = _parse_filters(filters)

    if bed_file:
        regions, bed_chrom = _load_bed_file(bed_file)
        mask = _bed_regions_to_mask(regions)
        masked = True
    else:
        bed_chrom = None
        masked = False

    if interval is not None:
        intervaled = True 
    else:
        intervaled = False

    if anc_seq_file is not None and use_AA is True:
        raise ValueError('You cannot use both `anc_seq_file` and `use_AA`')
    if anc_seq_file:
        ancestral_sequence = _load_fasta_file(anc_seq_file)
        has_anc_seq = True
    else:
        if use_AA:
            has_anc_seq = False
        else:
            has_anc_seq = False
            _print_out(
                _current_time(),
                'Polarizing with ``REF`` alleles: output SFS should be folded'
            )

    if vcf_file.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open

    # Should the INFO field be inspected on each line?
    if use_AA or 'INFO' in filters:
        need_info = True
    else:
        need_info = False

    tally = defaultdict(lambda: defaultdict(int))
    vcf_chrom = None
    counter = 0
    
    with openfunc(vcf_file, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                # Gather sample IDs
                split_header = line.split()
                sample_ids = split_header[9:]
                if len(sample_ids) == 0:
                    raise ValueError('VCF has zero samples')
                if pop_file is not None and pop_mapping is not None:
                    raise ValueError(
                        'You cannot use both `pop_mapping` and `pop_file`'
                    )
                elif pop_file is not None:
                    pop_mapping = _load_pop_file(pop_file)
                    if pops is not None:
                        pop_mapping = {pop: pop_mapping[pop] for pop in pops}
                elif pop_mapping is None:
                    _print_out(
                        _current_time(),
                        'No populations given: placing all samples in "ALL"'
                    )
                    pop_mapping = {'ALL': sample_ids}
                # Form a mapping from population IDs to sample indices
                pop_idx = defaultdict(list)
                for pop in pop_mapping:
                    for sample_id in pop_mapping[pop]:
                        if sample_id not in sample_ids:
                            raise ValueError(
                                f'Sample "{sample_id}" is absent from VCF'
                            )
                        pop_idx[pop].append(sample_ids.index(sample_id))
                continue
            
            # Print a status report
            if verbose > 0:
                if counter % verbose == 0 and counter > 1:
                    _print_out(
                        _current_time(),
                        f'Parsed site {pos}, line {counter}'
                    )
            counter += 1
            
            split_line = line.split()

            # Verify the consistency of chromosome labels
            chrom = split_line[0]
            if vcf_chrom is None:
                if bed_chrom is not None:
                    if not _check_chromosomes(bed_chrom, chrom):
                        raise ValueError('BED and VCF chromosomes mismatch')
                vcf_chrom = chrom 
            else:
                if chrom != vcf_chrom:
                    raise ValueError('VCF file must record only one chromosome')

            # Check whether ``POS`` is in `interval`, if given
            pos = int(split_line[1])
            pos0 = pos - 1
            if intervaled:
                if pos < interval[0]: 
                    stats['below_interval'] += 1
                    continue
                # Quit the loop if the end of `interval` has been passed
                if pos >= interval[1]:
                    _print_out(
                        _current_time(),
                        f'Quitting at site {pos}: beyond interval end'
                    )
                    break

            # Check whether ``POS`` is in an unmasked interval
            if masked:
                # Quit the loop if beyond the last unmasked site
                if pos0 >= len(mask):
                    _print_out(
                        _current_time(), 
                        f'Quitting at site {pos}: beyond mask end'
                    )
                    break
                elif mask[pos0] == True:
                    stats['mask_failed'] += 1
                    continue

            # Skip non-SNPs
            ref = split_line[3]
            alts = split_line[4].split(',')
            alleles = [ref] + alts
            skip = False
            for allele in alleles:
                if len(allele) > 1:
                    skip = True
                    break
            if skip:
                if stats['non_SNP_skipped'] == 0 and verbose:
                    _print_err(
                        _current_time(), f'Skipping site {pos}: non-SNP'
                    )
                stats['non_SNP_skipped'] += 1
                continue

            # Check whether the site is multiallelic
            if len(alts) > 1:
                if not allow_multiallelic:
                    if stats['multiallelic_skipped'] == 0 and verbose:
                        _print_err(
                            _current_time(),
                            f'Skipping site {pos}: multiallelic'
                        )
                    stats['multiallelic_skipped'] += 1
                    continue
                else:
                    stats['multiallelic_sites'] += 1

            # Build a dict representation of ``INFO`` if it will be needed
            if need_info:
                info_dict = _build_info_dict(split_line[7])

            # Assign the ancestral allele
            if has_anc_seq:
                if pos0 > len(ancestral_sequence):
                    raise ValueError(f'Site {pos} falls beyond FASTA sequence')
                anc = ancestral_sequence[pos0]
            elif use_AA:
                # Skip the site if `use_AA` and ``INFO/AA`` is absent
                if 'AA' not in info_dict:
                    warnings.warn('Absent ``INFO/AA`` field')
                    if stats['missing_AA_skipped'] == 0 and verbose:
                        _print_err(
                            _current_time(), f'Skipping site {pos}: '
                            'absent ``INFO/AA``'
                        )
                    stats['missing_AA_skipped'] += 1
                    continue
                anc = info_dict['AA']
            else:
                anc = ref

            # Check the validity of the ancestral allele
            skip, anc = _check_anc(anc, allow_low_confidence, pos=pos)
            if skip:
                continue

            # Skip the site if `anc` is not ``REF`` or ``ALT`` and multiallelic
            # sites are forbidden
            if not allow_multiallelic:
                if anc not in alleles:
                    if stats['unrepresented_anc_skipped'] == 0 and verbose:
                        _print_err(
                            _current_time(), f'Skipping site {pos}: '
                            'unrepresented ancestral allele'
                        )
                    stats['unrepresented_anc_skipped'] += 1
                    continue

            # Perform line-level filtering
            if 'QUAL' in filters:
                qual = split_line[5]
                if _filter_qual(qual, filters['QUAL'], pos=pos):
                    continue

            if 'FILTER' in filters:
                fltr = split_line[6]
                if _filter_filter(fltr, filters['FILTER'], pos=pos):
                    continue

            if 'INFO' in filters:
                if _filter_info(info_dict, filters['INFO'], pos=pos):
                    continue
 
            # Perform sample-level filtering and count alleles
            derived_idx = [alleles.index(a) for a in alleles if a != anc]
            frmt = split_line[8]
            split_frmt = frmt.split(':')
            samples = split_line[9:]
            pop_counts = defaultdict(list)
            for pop in pop_idx:
                split_samples = [samples[i].split(':') for i in pop_idx[pop]]
                if 'SAMPLE' in filters:
                    sample_dicts = [
                        dict(zip(split_frmt, s)) for s in split_samples
                    ]
                    for sample in sample_dicts:
                        if _filter_sample(sample, filters['SAMPLE'], pos=pos):      
                            sample['GT'] = '.'
                    GT_str = ''.join(s['GT'] for s in sample_dicts)
                else:
                    GT_idx = split_frmt.index('GT')
                    GT_str = ''.join([s[GT_idx] for s in split_samples])
                pop_counts[pop] = [
                    GT_str.count(str(i)) for i in range(len(alleles))
                ]
            # Count derived alleles and increment `tally`
            num_copies = tuple([sum(pop_counts[pop]) for pop in pop_idx])
            for i in derived_idx:
                num_derived = tuple([pop_counts[pop][i] for pop in pop_idx])
                tally[num_copies][num_derived] += 1

            stats['sites_passed'] += 1

    _print_out(_current_time(), f'Finished parsing {counter} sites')
    if len(stats) > 0:
        _print_out(_current_time(), 'Statistics:')
        for key in stats:
            print(f'{key}:\t{stats[key]}')

    if sum([tally[n][x] for n in tally for x in tally[n]]) == 0:
        warnings.warn('Parsed tally sums to 0')

    # Build the output data structure
    data = {}
    data['tally'] = tally
    data['pop_ids'] = [pop_id for pop_id in pop_idx]
    data['sample_sizes'] = {pop: ploidy * len(pop_idx[pop]) for pop in pop_idx}

    if return_stats:
        return data, copy(stats)
    
    return data


def _parse_filters(filters):
    """
    Parse the `filters` passed to `_tally_vcf` and create a nested dictionary
    for internal use. Valid key-value pairs are described below.

    There are basically two types of filters that can be imposed: numerical
    and categorical. Numerical filters give minimum values to fields, while
    categorical filters give a set of strings to which values must belong to 
    pass the filter.
    
    'QUAL' is numeric and maps to a minimum value for the ``QUAL`` field.
    'FILTER' is categorical.
    'INFO' and 'SAMPLE' may be numerical or categorical. The constistency of
        the field and its type is not checked here- e.g. specifying a 
        categorical filter for ``INFO/DP``, a numeric field, will not raise an
        error. 

    :param filters: Dictionary of filter variables to apply to a VCF file. 
    :type filters: dict

    :returns: Filters dictionary prepared for internal use.
    :rtype: defaultdict
    """
    nested_filters = defaultdict(dict)

    if filters is None:
        return nested_filters
    
    for key in filters:
        value = filters[key]
        split_key = key.split('/')
        if len(split_key) == 1:
            field = split_key[0]
            if field not in ('QUAL', 'FILTER'):
                raise ValueError(f'Invalid field ``{key}``')
            if field == 'QUAL':
                if type(value) not in (int, float):
                    raise TypeError('``QUAL`` must be a number')
                value = float(value)
            if field == 'FILTER':
                if type(value) is str:
                    value = set((value,))
                else:
                    value = set(value)
                for elem in value:
                    if type(elem) is not str:
                        raise TypeError('``FILTER`` must be str')
                    if elem == '.':
                        raise ValueError('"." cannot be used as a filter')
            nested_filters[field] = value
        elif len(split_key) == 2:
            field, subfield = split_key
            if field == 'FORMAT':
                field = 'SAMPLE'
            if field not in ('INFO', 'SAMPLE'):
                raise ValueError(f'Invalid key ``{key}``')
            try:
                value = float(value)
            except:
                if type(value) is str:
                    value = set((value,))
                else:
                    value = set(value)
                for elem in value:
                    if type(elem) is not str:
                        raise TypeError(f'Categorical ``{key}`` must be str')
                    if elem == '.':
                        raise ValueError('"." cannot be used as a filter')
            nested_filters[field][subfield] = value
        else:
            raise ValueError(f'Invalid key ``{key}``')

    return nested_filters


def _check_anc(anc, allow_low_confidence, pos=None):
    """
    Check whether the str `anc` encodes a valid ancestral state. Lines with
    invalid states (whether missing, not a valid nucleotide code, etc.) are
    skipped. 

    :type anc: str
    :type allow_low_confidence: bool
    :type pos: int, optional

    :returns: True if the line should be skipped due to `anc` invalidity,
        False otherwise. Also returns a copy of the ancestral allele string, 
        capitalizing it if it was lower case and `allow_low_confidence` was True
    """        
    skip = False
    ret_anc = None
    if anc not in ('A', 'T', 'C', 'G'):
        if anc == '.':
            warnings.warn('Missing ancestral allele')
            if stats['AA_missing'] == 0:
                _print_err(
                    _current_time(), f'Skipping site {pos}: '
                    'missing ancestral allele'
                )
            stats['AA_missing'] += 1
            skip = True
        elif anc in ('a', 't', 'c', 'g'):
            if allow_low_confidence:
                anc = anc.capitalize()
                stats['AA_low_confidence'] += 1
                ret_anc = anc.capitalize()
            else:
                if stats['AA_low_confidence_skipped'] == 0:
                    _print_err(
                        _current_time(), f'Skipping site {pos}: '
                        'low-confidence ancestral allele'
                    )
                stats['AA_low_confidence_skipped'] += 1
                skip = True
        else:
            warnings.warn('Invalid ancestral allele')
            if stats['AA_invalid'] == 0:
                _print_err(
                    _current_time(), f'Skipping site {pos}: '
                    'invalid ancestral allele'
                )
            stats['AA_invalid'] += 1
            skip = True
    else:
        ret_anc = anc

    return skip, ret_anc


def _filter_qual(qual, threshold, pos=None):
    """
    Apply a threshold to a ``QUAL`` column entry, printing soft error messages
    if the entry is missing or invalid.

    :type qual: str
    :type threshold: float
    :type pos: int, optional

    :returns: True if the site fails the QUAL filter and should be skipped,
        otherwise False.
    :rtype: bool
    """
    skip = False
    if qual == '.':
        if stats['QUAL_missing'] == 0:
            _print_err(_current_time(), f'Missing ``QUAL`` at site {pos}')
        stats['QUAL_missing'] += 1
    elif not qual.isnumeric():
        if stats['QUAL_invalid'] == 0:
            _print_err(_current_time(), f'Invalid ``QUAL`` at site {pos}')
        stats['QUAL_invalid'] += 1
    else:
        if float(qual) < threshold:
            stats['QUAL_failed'] += 1
            skip = True
        
    return skip


def _filter_filter(fltr, passing, pos=None):
    """
    Apply a filter to the a ``FILTER`` column entry. 

    :type fltr: str
    :type passing: set
    :type pos: int, optional

    :returns: True if the site fails the ``FILTER`` filter and should be 
        skipped, else False.
    :rtype: bool
    """
    skip = False
    if fltr == '.':
        if stats['FILTER_missing'] == 0:
            _print_err(_current_time(), f'Missing ``FILTER`` at site {pos}')
        stats['FILTER_missing'] += 1
    else:
        if fltr not in passing:
            stats['FILTER_failed'] += 1
            skip = True

    return skip


def _filter_info(info, info_filters, pos=None):
    """
    Apply `info_filters` to a dictionary representation of an ``INFO`` column
    entry. Increments appropriate statistics and raises warnings when fields 
    are absent.
    
    :type info: dict
    :type info_filters: dict
    :type pos: int, optional
    
    :returns: True if the site fails at least one ``INFO`` filter and should be
        skipped, False otherwise.
    :rtype: bool
    """
    skip = False
    for field in info_filters:
        if field not in info:
            warnings.warn(f"Absent ``INFO/{field}`` field")
            if stats[f'INFO/{field}_absent'] == 0:
                _print_err(
                    _current_time(), f'Absent ``INFO/{field}`` at site {pos}'
                )
            stats[f'INFO/{field}_absent'] += 1
        elif info[field] == '.':
            if stats[f'INFO/{field}_missing'] == 0:
                _print_err(
                    _current_time(), f'Missing ``INFO/{field}`` at site {pos}'
                )
            stats[f'INFO/{field}_missing'] += 1
        else:
            if type(info_filters[field]) == float:
                if float(info[field]) < info_filters[field]:
                    stats[f'INFO/{field}_failed'] += 1
                    skip = True
                    break
            else:
                if info[field] not in info_filters[field]:
                    stats[f'INFO/{field}_failed'] += 1
                    skip = True
                    break
    return skip


def _filter_sample(sample, sample_filters, pos=None):
    """
    Apply a `sample_filters` to a dictionary representation of a sample. 
    Returns True if the sample fails one or more filters and False else. Also
    gathers statistics and raises appopriate warnings when fields are absent.

    :type sample: dict
    :type sample_filters: dict
    :type pos: int, optional
    
    :returns: True if the sample fails at least one filter and should be
        skipped, False otherwise.
    :rtype: bool
    """
    skip = False
    for field in sample_filters:
        if field not in sample:
            warnings.warn(f"Absent ``SAMPLE/{field}`` field")
            if stats[f'SAMPLE/{field}_absent'] == 0:
                _print_err(
                    _current_time(), f'Absent ``SAMPLE/{field}`` at site {pos}'
                )
            stats[f'SAMPLE/{field}_absent'] += 1
        elif sample[field] == '.':
            if stats[f'SAMPLE/{field}_missing'] == 0:
                _print_err(
                    _current_time(), f'Missing ``SAMPLE/{field}`` at site {pos}'
                )
            stats[f'SAMPLE/{field}_missing'] += 1
        else:
            if type(sample_filters[field]) == float:
                if float(sample[field]) < sample_filters[field]:
                    stats[f'SAMPLE/{field}_failed'] += 1
                    skip = True
                    break
            else:
                if sample[field] not in sample_filters[field]:
                    stats[f'SAMPLE/{field}_failed'] += 1
                    skip = True
                    break
    return skip


def _build_info_dict(info_str):
    """
    Build a dictionary representation of the ``INFO`` field. When an item is 
    not a key=value pair, it is mapped to itself. For instance, 'GQ=30;X' 
    returns {'GQ': '30', 'X': 'X'}.
    """
    split_info = [field.split('=') for field in info_str.split(';')]
    info_dict = {}
    for item in split_info:
        if len(item) == 2:
            key, value = item
            info_dict[key] = value
        elif len(item) == 1:
            item = item[0]
            info_dict[item] = item
        else:
            raise ValueError(f'Unsupported format in ``INFO/{item}``')
        
    return info_dict


def _check_chromosomes(chrom1, chrom2):
    """
    Check whether two chromosome numbers match.

    :returns: True if the chromosome numbers match, False otherwise.
    :rtype: bool
    """
    _chrom1 = str(chrom1).lstrip("chr")
    _chrom2 = str(chrom2).lstrip("chr")
    if _chrom1 == _chrom2:
        ret = True 
    else:
        ret = False
    
    return ret


def _spectrum_from_tally(data, sample_sizes=None, mask_corners=True):
    """
    From a nested dictionary of derived allele count tallies emitted by the
    `_tally_vcf`, build an unfolded SFS. 
    
    If `sample_sizes` is given, then the output SFS will be projected to the
    specified sizes. Otherwise the sample sizes recorded in `data` are taken
    to determine its shape.
    
    :param tally: A dictionary representation of allele counts, loaded from a 
        VCF file with `_tally_vcf`. 
    :type tally: dict
    :param sample_sizes: The sample size of the output SFS (default None), 
        specified as a dictionary mapping population IDs to haploid sample 
        sizes. These must be less than or equal to the total sample sizes from
        VCF parsing.
    :type sample_sizes: dict, optional
    :param mask_corners: If True (default), mask the 'observed in no samples'
        and 'observed in all samples' entries of the returned SFS.
    :type mask_corners: bool, optional

    :returns: The SFS represented as a moments.Spectrum instance 
    :rtype: moments.Spectrum
    """
    def build_fs(_sizes):
        """
        Construct a single SFS with shape `_size`.
        """
        arr = np.zeros([n + 1 for n in _sizes])
        for entry in tally[_sizes]:
            arr[entry] += tally[_sizes][entry]
        fs = Spectrum_mod.Spectrum(
            arr, mask_corners=mask_corners, pop_ids=pop_ids
        )
        return fs
    
    def empty_fs(_sizes):
        """
        Construct an SFS with no entries and shape `_size`.
        """
        arr = np.zeros([n + 1 for n in _sizes])
        fs = Spectrum_mod.Spectrum(
            arr, mask_corners=mask_corners, pop_ids=pop_ids
        )
        return fs

    tally = data['tally']
    if len(tally) == 0:
        raise ValueError('Input data is empty')
    pop_ids = data['pop_ids']
    data_sizes = data['sample_sizes']
    keys = list(tally.keys())

    if sample_sizes is None:
        sample_sizes = data_sizes
        size_tuple = tuple([sample_sizes[pop] for pop in pop_ids])
        fs = build_fs(size_tuple)
    else: 
        # Check to make sure all sample_sizes <= data_sizes
        if not np.all([sample_sizes[p] <= data_sizes[p] for p in pop_ids]):
            raise ValueError(
                'One or more `sample_sizes` exceeds the sample size of data'
            )
        size_tuple = tuple([sample_sizes[pop] for pop in pop_ids])
        # Find records with sample sizes >= `size`
        valid_keys = []
        for key in keys:
            if np.all([m >= n for m, n in zip(key, size_tuple)]):
                valid_keys.append(key)
        if size_tuple in valid_keys:
            fs = build_fs(size_tuple)
        else:
            fs = empty_fs(size_tuple)
        # Project other valid sample sizes down to match the given one
        for key in valid_keys:
            if key == size_tuple:
                continue
            pfs = build_fs(key)
            fs += pfs.project(size_tuple)

    return fs


def _load_fasta_file(fasta_file):
    """
    Load a nucleotide sequence stored in FASTA format. Lines beginning with 
    '>' are treated as comments and are skipped.

    :param fasta_file: Pathname of the FASTA file to load.
    :type fasta_file: str

    :returns: A string representation of FASTA file contents.
    :rtype: str
    """
    if fasta_file.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    sequence_list = []
    with openfunc(fasta_file, "rb") as fin: 
        for lineb in fin:
            line = lineb.decode()
            if line.startswith(">"):
                continue 
            sequence_list.append(line.strip())
    sequence = ''.join(sequence_list)

    return sequence


def _load_bed_file(bed_file):
    """
    Load regions from a BED file as an array. Expects the structure 
        CHROM\tSTART\tEND...\n
    on each line, and skips any comment/header lines that begin with '#'. 
    Raises an error if the BED file has more than one unique CHROM entry.

    :param bed_file: Pathname of the BED file to load. 
    :type bed_file: str

    :returns: ndarray of BED file regions, BED chromosome ID
    :rtype: np.ndarray, str
    """
    if bed_file.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    chroms = []
    starts = []
    ends = []
    with openfunc(bed_file, "rb") as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('#'):
                continue
            split_line = line.split()
            chroms.append(split_line[0])
            starts.append(float(split_line[1]))
            ends.append(float(split_line[2]))
    chrom_set = set(chroms)
    # check that there is one unique CHROM
    if len(chrom_set) > 1:
        raise ValueError('BED files must describe one chromosome only')
    # check to make sure one or more lines were read
    elif len(chrom_set) == 0:
        raise ValueError('BED file has no valid contents')
    chrom = list(chrom_set)[0]
    regions = np.array(
        [[start, end] for start, end in zip(starts, ends)], dtype=np.int64
    )

    return regions, chrom


def _bed_regions_to_mask(regions):
    """
    Build a mask array from `regions` loaded from a BED file. The array equals 
    False within intervals in `regions` and True elsewhere outside them. Its
    length is equal to the highest region end position in `regions`.

    :param regions: Array of region starts and ends.
    :type regions: np.ndarray

    :returns: Boolean mask array representation of `regions`.
    :rtype: np.ndarray
    """
    L = regions[-1, 1]
    mask = np.ones(L, dtype=bool)
    for (start, end) in regions:
        mask[start:end] = False

    return mask


def _load_pop_file(pop_file):
    """
    Load a file that maps sample IDs to populations, formatted as
    SAMPLE_NAME POP_NAME
    where the delimiter can be any whitespace. Sample IDs should all be unique 
    and one sample/population pair should be specified per line. 
    
    :param pop_file: Population file formatted as decribed above.
    :type pop_file: str

    :returns: Dictionary that maps population names to lists of samples IDs.
    :rtype: dict
    """
    pop_mapping = defaultdict(list)
    with open(pop_file, "r") as fin:
        for line in fin:
            if line == "\n":
                continue 
            split_line = line.strip().split()
            if len(split_line) != 2:
                raise ValueError('Invalid `pop_file` format')
            sample_id, pop_id = split_line
            pop_mapping[pop_id].append(sample_id)
    # Check sample uniqueness
    all_samples = [pop_mapping[key][i] for key in pop_mapping 
                   for i in range(len(pop_mapping[key]))]
    if len(all_samples) != len(set(all_samples)):
        raise ValueError('All sample IDs in `pop_file` must be unique')

    return pop_mapping


def _write_bed_file(filename, regions, chrom):
    """
    Write a BED file.

    :param filename: Pathname of output file. Should end in .bed or .bed.gz. 
    :type filename: str
    :param regions: Array of BED regions to save.
    :type regions: np.ndarray
    :param chrom: Chromosome number to use in the CHROM column. All regions are 
        assigned to the same chromosome. 
    :type chrom: str

    :returns: None
    """
    if filename.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open
    with openfunc(filename, 'wb') as fout:
        fout.write('#CHROM\tSTART\tEND\n'.encode())
        for start, end in regions:
            fout.write(f'{chrom}\t{start}\t{end}\n'.encode())
        
    return 
