 .. _sec_sfs_parsing:

.. jupyter-execute::
    :hide-code:

===============================
Parsing the SFS from a VCF file
===============================

As described in the `SFS section <sfs>`_, the site frequencey spectrum (SFS) is 
a multidimensional histogram that records the counts, frequencies or probabilities
of the observed counts of SNPs in one or more populations. Here we show how to use 
``moments.Spectrum.from_vcf`` to compute the SFS from a VCF file. 

Basic usage
-----------

The only argument required to compute the SFS is the path to the desired VCF.
The function ``moments.Spectrum.from_vcf`` returns an SFS array in the form of a
``moments.Spectrum`` instance. A minimal example of usage is:

.. code-block:: python

    import moments 

    vcf_file = "path/to/vcf_file.vcf.gz"
    fs = moments.Spectrum.from_vcf(vcf_file)


Specifying populations
----------------------

By default, all samples present in the VCF are collected into a single population ``"ALL"``.
There are two ways to group samples into multiple populations. We can specify a ``pop_mapping``,
a dictionary that directly maps population names to lists of VCF sample IDs:

.. code-block:: python

    pop_mapping = {"pop_A": ["id_0", "id_1", "id_2"], "pop_B": ["id_4", "id_5"]}
    fs = moments.Spectrum.from_vcf(vcf_file, pop_mapping=pop_mapping)

Or we can specify the path to a ``pop_file``, a whitespace-separated file following the 
pattern ``sample_id pop_id`` on each line. Optionally, we can parse only a subset of 
the populations listed in this file by passing a list of population names to the argument ``pops``. 
If nothing is passed to this argument, then all populations in the file will be included.

.. code-block::

    id_0 pop_A
    id_1 pop_A
    id_2 pop_A
    id_4 pop_B
    id_5 pop_B
    id_6 pop_C

.. code-block:: python

    pop_file = 'path/to/pop_file.txt'
    pops = ['pop_A', 'pop_B']
    fs = moments.Spectrum.from_vcf(vcf_file, pop_file=pop_file, pops=pops)

Any VCF samples that are not mapped to a population are ignored. 

Using ancestral allele information
----------------------------------

By default, reference alleles are interpreted as ancestral alleles. In this case, 
we will most likely wish to pass ``True`` to the argument ``folded`` to fold the
returned SFS, as there is no general correspondence between ancestral and reference 
alleles. We can provide inferred ancestral alleles for polarizing alleles in two different
ways. If the input VCF file has ``AA`` (ancestral allele) information in its ``INFO`` field,
we can pass ``True`` to ``use_AA`` to obtain ancestral allele assignments from this
subfield. Sites where ``AA`` is absent or has missing data are skipped and an alert 
message is printed at the first such site. 

Otherwise, we can pass the path to a FASTA-format file containing estimated ancestral alleles 
to ``anc_seq_file``. Sites with invalid or missing data are skipped, raising a single alert 
message as described above. Often, FASTA files represent low-confidence ancestral allele 
assignments with lower-case nucleotide codes. By default, sites assigned these are skipped
as though the data were missing, but the assignments may be taken as valid by passing 
``True`` to ``allow_low_confidence``. 

Biallelic and multiallelic sites
--------------------------------

By default, multiallelic sites are skipped. We can pass ``True``to ``allow_multiallelic``
to include derived alleles at multiallelic sites as distinct entries in the SFS. 
When we provide an ancestral sequence and ``allow_multiallelic`` is ``False``, 
biallelic sites where the reference and alternate alleles both differ from 
the ancestral allele are skipped, because these sites represent recurrent mutations and 
the relationship between the derived alleles is unclear. Conversely, when ``allow_multiallelic``
is ``True``, this exception is ignored and all the derived alleles at such sites are counted.

Using filters
-------------

It is often desirable to set quality thresholds or categorical requirements for
the inclusion of sites in the returned SFS. We can do this by passing a flat 
dictionary representing the desired quantitative/categorical filters to ``filters``.
Valid key-value combinations are listed here. ``"QUAL"`` specifies a lower bound 
on the VCF ``QUAL`` column and should map to a float or integer. ``"FILTER"``
should map to a string or list/set/tuple of strings. To pass, sites must have a ``FILTER``
entry equal to the value of ``"FILTER"`` if it is a string, or to one of its 
elements if it is a string, tuple or list.

``"INFO/SUBFIELD"`` imposes a filter on ``SUBFIELD`` in the ``INFO`` column. Its 
value may be a float or integer, in which case it imposes a minimum threshold on that 
entry. It may also map to a string or list/set/tuple of strings, with equivalent 
behavior to ``"FILTER"``. ``"SAMPLE/SUBFIELD"`` works in the same way, but imposes 
filters on individual samples rather than lines. The fields ``"SAMPLE"`` and ``"FORMAT"`` 
are equivalent and refer to subfields enumerated in the ``FORMAT`` VCF column. 
Their types and behavior are the same as for ``INFO`` subfields, but filtering occurs 
at the sample level. An arbitrary example is:

.. code-block:: python

    filter_dict = {
        "QUAL": 30,
        "FILTER": "PASS",
        "INFO/DP": 30,
        "INFO/DB": "DB"
        "FORMAT/GQ": 30
        "FORMAT/DP": 30
    }
    fs = moments.Spectrum.from_vcf(vcf_file, filters=filter_dict)

The types of filters are not explicitly checked for consistency with their definitions
in the VCF file, so care should be taken when specifying them. Inappropriately typed 
filters will generally raise errors. Lines or samples with absent fields/missing data 
are not skipped, but one-time alert messages are printed for each unique exception.

Projecting to a smaller sample size
-----------------------------------

We may wish to reduce the size the output SFS to reduce the space it occupies in 
memory, to make computing an expected SFS for the same shape faster, to allow sites 
where some samples are filtered out or missing to be retained in output, or for 
other reasons. We can accomplish this by passing a dictionary of desired haploid sample sizes to 
``sample_sizes``. Any VCF sites with exactly this number of observed alleles will be 
retained without alteration in the output SFS, and the SFS from all sites with 
sample-size configurations larger than ``sample_sizes`` will be projected down to 
match. The output SFS is a sum over these cases. Projection is a procedure for reducing 
the size of the SFS by summing over the possible subsamplings of an entry. An example 
usage with the population file shown above is:

.. code-block:: python

    sample_sizes = {"pop_A": 4, "pop_B": 2, "pop_C": 2}
    fs = moments.Spectrum.from_vcf(
        vcf_file, 
        pop_file=pop_file, 
        sample_sizes=sample_sizes
    )

Note that sample sizes can be equal to, but not greater than, the total haploid sample 
size of a population.

Specifying regions
------------------

We can subset parsing to a genomic window by using the ``interval`` argument, which 
should be a 2-list of integers. This interval should be one-indexed and half-open. 
Additionally, we provide a mask file in BED format with the argument ``bed_file``,
which will filter out sites that fall outside its region intervals.
``bed_file`` can be given alongside ``interval``, so that only sites which fall 
within a BED interval and within ``interval`` are parsed. Note that BED file
intervals are half-open and zero-indexed. Also note that ``moments.Spectrum.from_vcf`` 
does not support VCF files that contain sites from multiple chromosomes. An 
example where both arguments are passed is:

.. code-block:: python

    bed_file = "path/to/bed_file.bed.gz"
    interval = [1, 10000001]
    fs = moments.Spectrum.from_vcf(vcf_file, bed_file=bed_file, interval=interval)

***********
Computing L
***********

We can compute the effective sequence length corresponding to our SFS, :math:`L`,
with the ``moments.Parsing.compute_L`` function. Its only required argument is 
``bed_file``, the path to the BED file that was used to parse the SFS. We can also
give an ``interval``, restricting sites to a one-indexed, half-open interval. Also, 
if we used an ancestral sequence from an external FASTA file, it can be passed to 
``anc_seq_file``, with the interpretation of low-confidence allele assignements modulated 
by ``allow_low_confidence``. Providing a FASTA file will restrict sites counted in 
:math:`L` to those with inferred ancestral states. A maximal example is: 

.. code-block:: python

    bed_file = "path/to/bed_file.bed.gz"
    interval = [1, 10000001]
    anc_seq = "path/to/anc_seq_file.fa.gz"
    L = moments.Parsing.compute_L(
        bed_file, 
        interval=interval,
        anc_seq_file=anc_seq,
        allow_low_confidence=False
    )

******************
Bootstrapping data
******************

TODO: Currently, Misc.bootstrap() works with the data dict to create bootstrap
replicates. We should replace this function to work with independently parsed
"tally" dictionaries from different regions, and show some example code blocks
here.
