 .. _sec_sfs_parsing:

.. jupyter-execute::
    :hide-code:

===============================
Parsing the SFS from a VCF file
===============================

As discussed in the `SFS section <sfs>`_, the SFS is a multidimensional array
that records the densities or counts of different derived allele frequencies. 
Here we show how to use ``moments.Spectrum.from_vcf`` to compute an SFS from a VCF 
file. 

Basic usage
-----------

The only argument required to parse the SFS from a VCF file is the path to the file.
The function ``moments.Spectrum.from_vcf`` returns an SFS array as an instance of 
``moments.Spectrum``. A minimal example of usage is:

.. code-block:: python

    import moments 

    vcf_file = "path/to/vcf_file.vcf.gz"
    fs = moments.Spectrum.from_vcf(vcf_file)


Specifying populations
----------------------

By default, all samples in the VCF are collected into a single population ``"ALL"``.
There are two ways to group samples into multiple populations. We can pass a dictionary
to directly maps the names of populations to lists of VCF sample IDs to ``pop_mapping``:

.. code-block:: python

    pop_mapping = {"pop_A": ["sid_0", "sid_1", "sid_2"], "pop_B": ["sid_4", "sid_5"]}
    fs = moments.Spectrum.from_vcf(vcf_file, pop_mapping=pop_mapping)

Or we can write a whitespace-separated file encoding the same relation with the format 
``sample population`` and pass it as ``pop_file``:

.. code-block::

    sid_0 pop_A
    sid_1 pop_A
    sid_2 pop_A
    sid_4 pop_B
    sid_5 pop_B

Any VCF samples that aren't mapped to a population are ignored. 

Using ancestral allele information
----------------------------------

By default, reference alleles are interpreted as ancestral alleles. In this case, 
we most likely will wish to flag the argument ``folded`` as ``True`` to fold the
returned SFS, as there is no general correspondence between ancestral and reference 
alleles. There are several ways to provide inferred ancestral alleles. When the 
input VCF file has ``AA`` (ancestral allele) information in its ``INFO`` field,
we can set ``use_AA`` as ``True`` to obtain ancestral allele assignments from this
field. Sites where ``AA`` is absent or has missing data are skipped. 

Alternately, we can pass a FASTA-format file containing inferred ancestral states 
as ``anc_seq_file``. Sites with invalid or missing data are skipped. Often, these 
files represent low-confidence ancestral allele assignments with lower-case nucleotide 
codes. By default sites assigned these are skipped, but the assignments may be taken 
as valid by passing ``True`` to ``allow_low_confidence``. 

Biallelic and multiallelic sites
--------------------------------

By default, multiallelic sites are skipped. We can pass ``True``to ``allow_multiallelic``
to include derived alleles at multiallelic sites as distinct entries in the SFS. 
When we provide an ancestral sequence and ``allow_multiallelic`` is ``False``, 
biallelic sites where the reference and alternate alleles both differ from 
the ancestral allele are skipped, because these sites represent recurrent mutations.
Conversely, when that argument is ``True``, this exception is ignored and all the derived
alleles at such sites are counted.

Using filters
-------------

It is often desirable to set quality thresholds or categorical requirements for
the inclusion of sites in the returned SFS. This is handled by passing a flat 
dictionary representing the desired quantitative/categorical requirements to ``filters``.
Valid keys-value combinations are listed here. ``"QUAL"`` specifies a lower bound 
on the VCF column of the same name and should map to a float or integer. ``"FILTER"``
should map to a string or list or strings. Sites whose ``FILTER`` value does not match
the dictionary entry (when a string) or one of its members (when a list) are filtered.

``"INFO/SUBFIELD"`` imposes a filter on ``SUBFIELD`` in the ``INFO`` column. When 
it maps to a float or integer, sites with lower values are filtered. It may also 
be mapped to a string or list of strings, with the same behavior as for the ``"FILTERS"``
field. ``"SAMPLE/SUBFIELD"`` works in the same way, but imposes filters on individual 
samples rather than lines. The fields ``"SAMPLE"`` and ``"FORMAT"`` are equivalent 
and may be used equivalently to refer to sample-specific fields enumerated in the 
``FORMAT`` VCF column. An arbitrary and comprehensive example is:

.. code-block:: python

    filter_dict = {
        "QUAL": 30,
        "FILTER": "PASS",
        "INFO/DP": 30,
        "INFO/DB": "DB"
        "SAMPLE/GQ": 30
    }
    fs = moments.Spectrum.from_vcf(vcf_file, filters=filter_dict)

The types of filters are not explicitly checked for consistency with their definitions
in the VCF file, so care should be taken when specifying them.
Lines or samples with absent fields/missing data are not filtered out, but one-time
alert messages are printed for each unique exception.

Projecting to a smaller sample size
-----------------------------------

Specifying regions
------------------

We can subset parsing to a genomic window using the ``interval`` argument, which 
should be a 2-list of integers. This interval should be one-indexed and half-open. 
Additionally, we can mask sites by providing a BED file with the argument ``bed_file``,
which will filter out sites that fall outside its region intervals.
``bed_file`` can be given alongside ``interval``, so that only sites which fall 
within a BED interval and within ``interval`` are parsed. Note that BED file
intervals are half-open and zero-indexed. Also note that ``moments.Spectrum.from_vcf`` 
does not support VCF files that contain sites from multiple chromosomes. Concretely,

.. code-block:: python

    import moments 

    bed_file = "path/to/bed_file.bed.gz"
    interval = [1, 10000001]
    fs = moments.Spectrum.from_vcf(vcf_file, bed_file=bed_file, interval=interval)

***********
Computing L
***********

******************
Bootstrapping data
******************

TODO: Currently, Misc.bootstrap() works with the data dict to create bootstrap
replicates. We should replace this function to work with independently parsed
"tally" dictionaries from different regions, and show some example code blocks
here.
