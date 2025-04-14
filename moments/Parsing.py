# placeholder
"""
Functions for computing the SFS and related quantities from sequence data. 
"""

from collections import defaultdict
import gzip
import numpy as np
import os
import sys
import time
import warnings

from moments import Spectrum


def parse_vcf(
    vcf_file,
    pop_mapping=None,
    pop_file=None,
    bed_file=None,
    interval=None,
    anc_seq_file=None,
    allow_low_confidence=True,
    use_AA=True,
    folded=False,
    filters=None,
    allow_multiallelic=False,
    sample_sizes=None,
    mask_corners=True,
    ploidy=2
):
    """
    Compute the SFS from genotype data stored in a .vcf file.



    When `anc_seq_file` is not given and `use_AA` is False, the reference allele 
    is taken as the ancestral allele and a warning is raised- the SFS should be
    folded after the fact or using `folded` in this case.
    
    :param vcf_file: Pathname of the .vcf file to parse. The file may be 
        gzipped/bgzipped or uncompressed.
    :type vcf_file: str
    :param pop_file: Pathname of a whitespace-separated file mapping samples
        to populations with the format SAMPLE POPULATION. Sample names must
        be unique and there should be one of them on each line (default None 
        combines all samples into a single population). 
    :type pop_file: str
    :param bed_file: Pathname of a .bed file defining the regions to parse;
        useful for applying masks to exclude difficult-to-call or functionally
        constrained regions (default None). 
    :type bed_file: str
    :param interval: 2-tuple or 2-list specifying the (inclusive, 0-indexed) 
        first and (exclusive, 0-indexed) last positions to parse; defines a
        genomic window (default None parses the whole .vcf file).
    :type interval: tuple or list of integers
    :param anc_seq_file: Pathname of a .fa file containing estimated ancestral 
        nucleotide states (default None).
    :type anc_seq_file: str
    :param allow_low_confidence: If True, skip sites without high-confidence 
        ancestral state assignments in a given .fa file, which are denoted by 
        capitalized nucleotide codes (default True). 
    :type skip_low_conf: bool
    :param use_AA: If True, use entries in the .vcf ``INFO/AA`` to polarize
        the ancestral allele (default True). If this field is missing 
        altogether, raises an error; if data is missing for a sites, skips the
        site and raises a warning.
    :type use_AA: bool
    :param folded: If True, return the folded SFS (default False).
    :type folded: bool
    :param filters: Dictionary specifying minimum values for one 
        or more fields QUAL, FILTER, INFO/DP, INFO/GQ etc (default None).
        TODO WIP
    :type filters: dict
    :param allow_multiallelic: If True, tally derived alleles at multiallelic
        sites.
    :type allow_multiallelic: bool
    :param sample_sizes: Dictionary assigning minimum sample sizes for each
        population- sites with more alleles than specified are projected to
        given sizes (default None).
    :type sample_sizes: dict

    :returns: ``moments.Spectrum`` instance holding the parsed SFS.
    """
    data = _tally_vcf(
        vcf_file,
        pop_mapping=pop_mapping,
        pop_file=pop_file,
        bed_file=bed_file,
        interval=interval,
        anc_seq_file=anc_seq_file,
        allow_low_confidence=allow_low_confidence,
        use_AA=use_AA,
        filters=filters,
        allow_multiallelic=allow_multiallelic,
        return_stats=False,
        print_stats=True,
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


warned = defaultdict(bool)
stats = defaultdict(int)


def _clear_stats():
    """
    Re-initialize `warned` and `stats`.
    """
    global warned, stats
    warned = defaultdict(bool)
    stats = defaultdict(int)

    return


def _warn(warn_flag, warn_message):
    """
    If warning `warn_flag` has not yet been raised, raise it with warning 
    message `warn_message`. Also increment `stats['warn_flag']`. 

    :type warn_flag: str
    :type warn_message: str

    :returns: None
    """
    if not warned[warn_flag]:
        warnings.warn(warn_message)
        warned[warn_flag] = True
    stats[warn_flag] += 1

    return


def _tally_vcf(
    vcf_file,
    pop_mapping=None,
    pop_file=None,
    bed_file=None,
    interval=None,
    anc_seq_file=None,
    allow_low_confidence=False,
    use_AA=False,
    filters=None,
    allow_multiallelic=False,
    return_stats=False,
    print_stats=True,
    ploidy=2
):
    """
    Read a dictionary representation of derived allele counts from a VCF file.

    This object maps haploid, per-population sample sizes (of which there can be 
    several- if  filters are applied at the SAMPLE level or there are missing 
    genotype data in the VCF) to nested dictionaries, which map derived allele 
    counts to their observed count in the VCF. This can be realized as the SFS 
    using `_spectrum_from_tally`. 

    There are several ways to specify the ancestral states used to polarize 
    alleles. The default behavior takes the ``REF`` allele specified in the VCF
    file as ancestral- when this is done, the SFS should be folded. If `use_AA`
    is True, the ``INFO/AA`` field in the VCF is used. Sites with absent 
    ``AA/INFO``, missing data or invalid values are skipped, raising a warning. 
    If `anc_seq_file` (FASTA format) is provided, ancestral states are read 
    from the given file. Any sites with undefined ancestral states 
    (e.g. "N", "-", ".") will be skipped, raising a warning. If 
    `allow_low_confidence` is False, then sites assigned with lower-case 
    nucleotide codes in the FASTA will also be skipped.

    :param vcf_file: Pathname of the VCF file from which to read. 
    :type vcf_file: str
    :param pop_mapping: Optional dictionary mapping string population IDs to
        lists of sample IDs present in the VCF file (default None). Mutually
        exclusive with `pop_file`. 
    :type pop_mapping: dict, optional
    :param pop_file: Optional pathname to file mapping sample IDs to population 
        IDs (default None), with line format SAMPLE POPULATION, where strings 
        are seperated by any whitespace.
    :type pop_file: str, optional
    :param bed_file: Optional pathname to a mask file- only sites within regions
        defined in the file will be tallied (default None).
    :type bed_file: str, optional
    :param interval: Optional positional interval within which to tally sites
        (default None), specified as a 2-list or 2-tuple.
    :type interval: list, optional
    :param anc_seq_file: Pathname of a FASTA file assigning ancestral states
        to sites (default None).
    :type anc_seq_file: str, optional
    :param allow_low_confidence: If True (default False) and `anc_seq_file` is
        given, allows low-confidence ancestral state assignments, represented 
        by lower-case nucleotide codes. If False, these sites are skipped.
    :type allow_low_confidence: bool, optional
    :param use_AA: If True, use the field ``INFO/AA`` specified in the VCF file
        to asign ancestral states (default False). An error is raised if this
        field is absent. 
    :type use_AA: bool, optional
    :param filters: A dictionary specifying filters on VCF lines or samples,
        with any combination of the following key-value pairs:
        'QUAL' is the minimum allowed ``QUAL`` score for a line and must be 
            numeric.
        'FILTER' may be a string or list of strings; lines with ``FILTER`` 
            entries not specified will be skipped.
        'FORMAT/FIELD' may be numeric (specifying a minimum value) or 
            categorical (a string or list of strings not to filter).
        'SAMPLE/FIELD' is like 'FORMAT/FIELD' but applies at the sample level.
        Many ``FORMAT`` and ``SAMPLE`` fields may be given.
        Raises warnings when targetted fields are missing.
    :type filters: dict
    :param allow_multiallelic: If True (default False), include sites with more
        than one alternate allele as separate entries in the tally. Also allow
        sites where neither alternate nor reference allele matches the assigned
        ancestral state.
    :type allow_multiallelic: bool, optional
    :param return_stats: If True (default False), return a dictionary of 
        statistics describing the number sites that were filtered out for 
        various reasons along with `tally`.
    :type return_stats: bool, optional
    :param print_stats: If True (default), print statistics on the number of 
        sites filtered out for various reasons upon completion.
    :type print_stats: bool, optional
    :param ploidy: Optional (default 2), defines the maximum derived allele 
        count in combination with the number of samples.
    :type plody: int, optional
     
    :return: Dictionary mapping sample sizes to dictionaries which map derived
        allele counts to the count of their occurences in the VCF file.
    :rtype: dict
    """
    # inspect `filters` 
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

    if anc_seq_file is not None and use_AA is not None:
        raise ValueError('You cannot give both `anc_seq_file` and `use_AA`')
    elif anc_seq_file:
        ancestral_sequence = _load_fasta_file(anc_seq_file)
        has_anc_seq = True
    elif use_AA:
        has_anc_seq = False
    else:
        has_anc_seq = False
        warnings.warn('Polarizing with ``REF`` alleles: SFS should be folded')
    
    if vcf_file.endswith('.gz'):
        openfunc = gzip.open 
    else:
        openfunc = open

    # should the INFO field be inspected on each line?
    if use_AA or 'INFO' in filters:
        need_info = True
    else:
        need_info = False

    _clear_stats()
    tally = defaultdict(lambda: defaultdict(int))
    vcf_chrom = None
    
    with openfunc(vcf_file, 'rb') as fin:
        for lineb in fin:
            line = lineb.decode()
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                # gather sample ids and populations
                split_header = line.split()
                sample_ids = split_header[9:]
                if len(sample_ids) == 0:
                    raise ValueError('VCF has zero samples')
                if pop_file is not None and pop_mapping is not None:
                    raise ValueError(
                        'You cannot provide both `pop_mapping` and `pop_file`'
                    )
                elif pop_file is not None:
                    pop_mapping = _load_pop_file(pop_file)
                elif pop_mapping is None:
                    warnings.warn(
                        'No population specification given: ' 
                        'placing all samples in population "ALL"'
                    )
                    pop_mapping = {'ALL': sample_ids}
                # form mapping from population IDs to sample indices
                pop_idx = defaultdict(list)
                for pop in pop_mapping:
                    for sample_id in pop_mapping[pop]:
                        if sample_id not in sample_ids:
                            raise ValueError(
                                f'Sample "{sample_id}" is absent from VCF'
                            )
                        pop_idx[pop].append(sample_ids.index(sample_id))
                continue
            
            split_line = line.split()

            # check chromosome agreement
            chrom = split_line[0]
            if vcf_chrom is None:
                if bed_chrom is not None:
                    if not _match_chromosomes(bed_chrom, chrom):
                        raise ValueError('BED and VCF chromosomes mismatch')
                vcf_chrom = chrom 
            else:
                if chrom != vcf_chrom:
                    raise ValueError('VCF file must record only one chromosome')

            # check interval (decrement position to make it 0-indexed)
            pos = int(split_line[1])
            pos0 = pos - 1
            if intervaled:
                if pos0 < interval[0]: 
                    stats['below_interval'] += 1
                    continue
                # quit the loop if end of interval has been passed
                if pos0 >= interval[1]:
                    warnings.warn('Beyond specified interval end: quitting')
                    break

            # check mask
            if masked:
                # quit the loop if beyond the last unmasked site
                if pos0 >= len(mask):
                    warnings.warn('Beyond specified mask end: quitting')
                    break
                elif mask[pos0] == True:
                    stats['mask_failed'] += 1
                    continue

            # skip non-SNPs
            ref = split_line[3]
            alts = split_line[4].split(',')
            alleles = [ref] + alts
            skip = False
            for allele in alleles:
                if len(allele) > 1:
                    skip = True
                    break
            if skip:
                _warn('non_SNP_skipped', f'Skipping non-SNP at {pos}')
                continue

            # check whether multiallelic
            if len(alts) > 1:
                if not allow_multiallelic:
                    _warn('multiallelic_skipped',
                          f'Skipping multiallelic site at {pos}')
                    continue
                else:
                    stats['multiallelic_sites'] += 1

            # build a dict representation of ``INFO`` if it will be needed
            if need_info:
                info_dict = _build_info_dict(split_line[7])

            # assign ancestral allele
            if has_anc_seq:
                if pos0 > len(ancestral_sequence):
                    raise ValueError(f'Site {pos} falls beyond FASTA sequence')
                anc = ancestral_sequence[pos0]
            elif use_AA:
                # skip site if AA is not present
                if 'AA' not in info_dict:
                    _warn('AA_absent', f'``INFO/AA`` is absent at {pos}')
                    continue
                anc = info_dict["AA"]
            else:
                anc = ref

            # check validity of ancestral allele
            if _check_anc(anc, allow_low_confidence, pos):
                continue
            
            # skip if `anc` is not REF/ALT and multiallelic sites are forbidden
            if not allow_multiallelic:
                if anc not in alleles:
                    _warn('AA_unrepresented',
                          f'Ancestral allele not represented in ``REF``, ``ALT`` at {pos}')
                    continue

            # perform line-level filtering
            if 'QUAL' in filters:
                qual = split_line[5]
                if _filter_QUAL(qual, filters['QUAL'], pos):
                    continue

            if 'FILTER' in filters:
                fltr = split_line[6]
                if _filter_FILTER(fltr, filters['FILTER'], pos):
                    continue

            if 'INFO' in filters:
                if _filter_INFO(info_dict, filters['INFO'], pos):
                    continue
 
            # perform sample-level filtering and count alleles
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
                        if _filter_sample(sample, filters['SAMPLE'], pos):      
                            sample['GT'] = '.'
                    GT_str = ''.join(s['GT'] for s in sample_dicts)
                else:
                    GT_idx = split_frmt.index('GT')
                    GT_str = ''.join([s[GT_idx] for s in split_samples])
                pop_counts[pop] = [
                    GT_str.count(str(i)) for i in range(len(alleles))
                ]
            # count derived alleles and increment tally
            num_copies = tuple([sum(pop_counts[pop]) for pop in pop_idx])
            for i in derived_idx:
                num_derived = tuple([pop_counts[pop][i] for pop in pop_idx])
                tally[num_copies][num_derived] += 1

    if print_stats:
        print('Statistics')
        for key in stats:
            print(f'{key}:\t{stats[key]}')

    if sum([tally[n][x] for n in tally for x in tally[n]]) == 0:
        warnings.warn('Parsed tally sums to 0')

    # build the output data structure
    data = {}
    data['tally'] = tally
    data['pop_ids'] = [pop_id for pop_id in pop_idx]
    data['sample_sizes'] = tuple(
        [ploidy * len(pop_idx[pop]) for pop in pop_idx]
    )

    if return_stats:
        return data, stats
    
    return data


def _parse_filters(filters):
    """
    Parse the `filters` passed to `_tally_vcf` and create a nested dictionary
    for internal use. Raises errors if any fields are not validly specified.

    :param filters: Dictionary mapping VCF fields to minumum values (when value
        is a number) or required categories (when a str, tuple or list) to 
        pass the filtering step and be included in the derived allele tally.
    :type filters: dict

    :returns: Filters dictionary prepared for internal use.
    :rtype: defaultdict
    """
    nested_filters = defaultdict(dict)
    if filters is not None:
        for key in filters:
            value = filters[key]
            split_key = key.split('/')
            if len(split_key) == 1:
                field = split_key[0]
                if field not in ('QUAL', 'FILTER'):
                    raise ValueError(f'Invalid field ``{key}``')
                if field == 'QUAL':
                    if type(value) not in (int, float, str):
                        raise TypeError('``QUAL`` must be a number')
                    if type(value) is str:
                        if value.isnumeric():
                            value = float(value)
                        else:
                            raise ValueError('``QUAL`` must be a number')
                    else:
                        value = float(value)
                if field == 'FILTER':
                    if type(value) not in (str, list, tuple):
                        raise TypeError('``FILTER`` must be categorical')
                    if type(value) is str:
                        value = (value,)
                    else:
                        value = tuple(value)
                    for elem in value:
                        if elem == '.':
                            raise ValueError('"." cannot be used as a filter')
                nested_filters[field] = value
            elif len(split_key) == 2:
                field, subfield = split_key
                if field not in ('INFO', 'FORMAT', 'SAMPLE'):
                    raise ValueError(f'Invalid key ``{key}``')
                # FORMAT indicates the same thing as SAMPLE
                if field == 'FORMAT':
                    field = 'SAMPLE'
                # numeric fields
                if type(value) in (int, float):
                    value = float(value)
                # categorical fields
                elif type(value) is str:
                    if value == '.':
                        raise ValueError('"." cannot be used as a filter')
                    value = (value,)
                elif type(value) in (list, tuple):
                    for elem in value:
                        if type(elem) is not str:
                            raise TypeError(
                                f'Categorical ``{key}`` must be str'
                            )
                        if elem == '.':
                            raise ValueError('"." cannot be used as a filter')
                    value = tuple(value)
                else:
                    raise TypeError(f'Invalid ``{key}`` type')
                nested_filters[field][subfield] = value
            else:
                raise ValueError(f'Invalid key ``{key}``')

    return nested_filters


def _check_anc(anc, allow_low_confidence, pos):
    """
    Check whether the str `anc` represents a valid ancestral state. Lines with
    invalid states (whether missing, not a valid nucleotide code, etc.) are
    skipped. 

    :type anc: str
    :type allow_low_confidence: bool
    :type pos: int

    :returns: True if the line should be skipped due to `anc` invalidity,
        False otherwise.
    """        
    skip = False
    if anc not in ('A', 'T', 'C', 'G'):
        if anc == '.':
            _warn('AA_missing', f'Ancestral allele is missing at {pos}')
            skip = True
        elif anc in ('a', 't', 'c', 'g'):
            if allow_low_confidence:
                anc = anc.capitalize()
                stats['AA_low_confidence'] += 1
            else:
                _warn('AA_low_confidence_skipped',
                      f'Skipping line with low-conf. ancestral allele at {pos}')
                skip = True
        else:
            _warn('AA_invalid', f'Ancestral allele is invalid at {pos}')
            skip = True

    return skip


def _filter_QUAL(qual, threshold, pos):
    """
    Called during VCF parsing if ``QUAL`` is being filtered. 
    
    Checks whether ``QUAL`` is missing ('.'), then whether it is numeric, then 
    whether it is greater than a specified minimum. If ``QUAL`` is missing or
    non-numeric, raises a warning but does not skip the site.

    :type qual: float
    :type threshold: float
    :type pos: int

    :returns: True if the site fails the QUAL filter and should be skipped,
        otherwise False.
    :rtype: bool
    """
    skip = False
    if qual == '.':
        _warn('QUAL_missing', f'``QUAL`` is missing at {pos}')
    elif not qual.isnumeric():
        _warn('QUAL_invalid', f'``QUAL`` is invalid at {pos}')
    else:
        if float(qual) < threshold:
            stats['QUAL_failed'] += 1
            skip = True
        
    return skip


def _filter_FILTER(fltr, passing, pos, warned, stats):
    """
    Checks whether the ``FILTER`` entry for a line (`fltr`) is missing, raising
    a warning if it is- then checks whether it matches an element of `passing`,
    returning True if it is not and the line should be skipped.

    :type fltr: str
    :type passing: tuple
    :type pos: int
    :type warned: defaultdict(bool)
    :type stats: defaultdict(int)
    
    :returns: True if the site fails the FILTER filter and should be skipped,
        else False.
    :rtype: bool
    """
    skip = False
    if fltr == '.':
        if not warned[f'FILTER_missing']:
            warnings.warn(f'``FILTER`` is missing at {pos}')
            warned[f'FILTER_missing'] = True
        stats[f'FILTER_missing'] += 1
    else:
        if fltr not in passing:
            stats['FILTER_failed'] += 1
            skip = True

    return skip


def _filter_INFO(info, info_filters, pos):
    """
    Check whether a line fails any filters on ``INFO``. If relevant fields are
    missing or completely absent, raises warnings. Filters can be either numeric
    or categorical. 
    
    :type info: dict
    :type info_filters: dict
    :type pos: int
    
    :returns: True if the site fails at least one INFO filter and should be
        skipped, False otherwise.
    :rtype: bool
    """
    skip = False
    for field in info_filters:
        if field not in info:
            _warn(f'INFO/{field}_absent', 
                  f'``INFO/{field}`` is absent at {pos}')
        elif info[field] == '.':
            _warn(f'INFO/{field}_missing',
                  f'``INFO/{field}`` is missing at {pos}')
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


def _filter_sample(sample, sample_filters, pos, warned, stats):
    """
        
    :type sample: dict
    :type sample_filters: dict
    :type pos: int
    :type warned: defaultdict(bool)
    :type stats: defaultdict(int)
    
    :returns: True if the sample fails at least one SAMPLE filter and should be
        skipped, False otherwise.
    :rtype: bool
    """
    skip = False
    for field in sample_filters:
        if field not in sample:
            _warn(f'SAMPLE/{field}_absent',
                  f'``SAMPLE/{field}`` is absent at {pos}')
        elif sample[field] == '.':
            _warn(f'SAMPLE/{field}_missing',
                  f'``SAMPLE/{field}`` is missing at {pos}')
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
    Build a dictionary representation of the ``INFO`` field. 
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
            raise ValueError(f'Unsupported format in ``INFO/{item}')
        
    return info_dict


def _match_chromosomes(chrom1, chrom2):
    """
    Check whether two chromosome numbers match.

    :returns: True if the chromosome numbers match, False otherwise.
    :rtype: bool
    """
    _chrom1 = str(chrom1).lstrip("chr")
    _chrom2 = str(chrom2).lstrip("chr")
    print(_chrom1, _chrom2)
    if _chrom1 == _chrom2:
        ret = True 
    else:
        ret = False
    
    return ret


def _spectrum_from_tally(
    data, 
    sample_sizes=None, 
    mask_corners=True, 
    pop_ids=None
):
    """
    From a nested dictionary of derived allele count tallies emitted by the
    `_tally_vcf`, build an SFS. 
    
    If `sample_sizes` or `pop_ids` are not given, then these values are taken
    from `tally`. If `sample_sizes` is given, then the output SFS will be this 
    size- all records with higher samples will be projected down to the given
    size and added to the output.

    If `sample_sizes` is larger than the size recorded in `tally`, an error is 
    raised.
    
    :param tally: A dictionary representation of allele counts loaded from a 
        VCF file from `tally_vcf`. 
    :type tally: dict
    :param sample_sizes: The sample size of the output SFS (default None). If 
        given, projects all records with equal or higher sample size into the 
        same output array. Must be less than or equal to the highest sample size
        recorded in `tally`.
    :type sample_sizes: tuple, optional
    :param mask_corners: If True (default), mask the 'observed in no samples'
        and 'observed in sall samples' entries of the SFS.
    :type mask_corners: bool, optional
    :param pop_ids: Optional list of population IDs associated to the SFS 
        (default None).
    :type pop_ids: list, optional

    :returns: Site frequency spectrum represented as a moments.Spectrum instance 
    :rtype: moments.Spectrum
    """
    def build_fs(_sizes):
        """
        Construct a single SFS with shape `_size`.
        """
        arr = np.zeros([n + 1 for n in _sizes])
        for entry in tally[_sizes]:
            arr[entry] += tally[_sizes][entry]
        fs = Spectrum(arr, mask_corners=mask_corners, pop_ids=pop_ids)
        return fs
    
    def empty_fs(_sizes):
        """
        Construct an SFS with no entries and shape `_size`.
        """
        arr = np.zeros([n + 1 for n in _sizes])
        fs = Spectrum(arr, mask_corners=mask_corners, pop_ids=pop_ids)
        return fs
    
    tally = data['tally']
    keys = list(tally.keys())

    if pop_ids is None:
        if 'pop_ids' in data:
            pop_ids = data['pop_ids']
        else:
            pop_ids = None
    
    if sample_sizes is None:
        if 'sample_sizes' in data:
            sample_sizes = data['sample_sizes']
        else:
            sample_sizes = tuple(
                [max([key[i] for key in keys]) for i in range(len(keys[0]))]
            )
        fs = build_fs(sample_sizes)
    else: 
        if type(sample_sizes) == list:
            sample_sizes = tuple(sample_sizes)
        # find records with sample sizes >= `size`
        valid_keys = []
        for key in keys:
            if np.all([m >= n for m, n in zip(key, sample_sizes)]):
                valid_keys.append(key)
        if sample_sizes in valid_keys:
            fs = build_fs(sample_sizes)
        else:
            fs = empty_fs(sample_sizes)
        # project other valid sample sizes down to the primary one
        for key in valid_keys:
            if key == sample_sizes:
                continue
            pfs = build_fs(key)
            fs += pfs.project(sample_sizes)

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
    If `anc_seq_file` is given, then only sites which are assigned an ancestral
    state are counted. Whether or not low-confidence assignments- conventionally
    represented with lower-case letters- are counted is modulated by 
    `allow_low_confidence`.

    :param bed_file: Pathname of a BED file specifying regions over which `L` 
        is to be computed.
    :param interval: 2-list specifying the interval over which to compute `L`. 
        The lower bound is inclusive, the upper noninclusive (default None).
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
        if end > len(mask):
            warnings.warn('Interval end is longer than mask')
        mask = mask[start:end]
    L = len(mask) - np.count_nonzero(mask)

    return L


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
    Load regions from a BED file. Expects the structure 
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
            starts.append(int(split_line[1]))
            ends.append(int(split_line[2]))
    chrom_set = set(chroms)
    # check that there is one unique CHROM
    if len(chrom_set) > 1:
        raise ValueError('BED files must describe one chromosome only')
    # check to make sure one or more lines were read
    elif len(chrom_set) == 0:
        raise ValueError('BED file has no valid contents')
    chrom = list(chrom_set)[0]
    regions = np.array([[start, end] for start, end in zip(starts, ends)])

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
    and one sample should be specified per line. 
    
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
    # check sample uniqueness
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


