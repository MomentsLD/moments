# Generates each chromosome's VCF independently, after simulating together and
# then combine them into a single VCF across all chromosomes. These will be
# used to check that Parsing gets the same statistics from each individual VCF
# compared to filtering by chromosome in the combined VCF.

import msprime
assert msprime.__version__ > "1.0", "Install msprime 1.0 or higher"

import moments
import numpy as np
import math
import os

L = 1e5
num_chrom = 4
u = r = 1e-8
r_break = math.log(2)
N = 1e4
n = 10
seed = 123

# set up recombination map following the documentation here:
# https://tskit.dev/msprime/docs/latest/ancestry.html#multiple-chromosomes

chrom_pos = [0]
map_pos = [0]
rates = []
for c in range(num_chrom):
    chrom_pos.append(chrom_pos[-1] + L)
    rates.append(r)
    rates.append(r_break)
    map_pos.append(chrom_pos[-1])
    map_pos.append(chrom_pos[-1] + 1)

map_pos = map_pos[:-1]
rates = rates[:-1]
rate_map = msprime.RateMap(position=map_pos, rate=rates)

# run simulations

ts = msprime.sim_ancestry(
    n,
    population_size=N,
    recombination_rate=rate_map,
    model=[
        msprime.DiscreteTimeWrightFisher(duration=100),
        msprime.StandardCoalescent(),
    ],
    random_seed=seed,
)

ts = msprime.sim_mutations(ts, rate=u, random_seed=seed)

print("num trees:", ts.num_trees)
print("num muts:", ts.num_mutations)

# write each VCF, combined and separate

ts_chroms = []
for j in range(len(chrom_pos) - 1):
    start, end = chrom_pos[j: j + 2]
    chrom_ts = ts.keep_intervals([[start, end]], simplify=False).trim()
    ts_chroms.append(chrom_ts)

print("num muts still:", sum([ts_c.num_mutations for ts_c in ts_chroms]))

for i, ts in enumerate(ts_chroms):
    with open(f"chr{i+1}.vcf", "w+") as fout:
        ts.write_vcf(fout, contig_id=f"{i+1}")

# combine the VCFs

with open("chrALL.vcf", "w+") as fout:
    for i in range(1, num_chrom + 1):
        with open(f"chr{i}.vcf", "r") as fin:
            for line in fin:
                if line.startswith("#") and i != 1:
                    continue
                else:
                    fout.write(line)

# set up input files

for i in range(1, num_chrom + 1):
    os.system(f"gzip chr{i}.vcf")

os.system("gzip chrALL.vcf")

with open("rec_map.txt", "w+") as fout:
    fout.write("Pos Map(cM)\n")
    fout.write("0 0\n")
    fout.write(f"{L} {r * L * 100}")

# check for equality in parsed statistic sums

ld_stats_sep = {}
ld_stats_all = {}

r_bins = [0, 1e-6, 1e-5, 1e-4]

for i in range(1, num_chrom + 1):
    print("parsing chromosome", i)
    ld_stats_sep[i] = moments.LD.Parsing.compute_ld_statistics(
        f"chr{i}.vcf.gz",
        bed_file="multichrom.bed",
        rec_map_file="rec_map.txt",
        r_bins=r_bins,
        report=False,
        use_h5=True,
    )
    ld_stats_all[i] = moments.LD.Parsing.compute_ld_statistics(
        "chrALL.vcf.gz",
        chromosome=i,
        bed_file="multichrom.bed",
        rec_map_file="rec_map.txt",
        r_bins=r_bins,
        report=False,
        use_h5=True,
    )

for i in range(1, num_chrom + 1):
    for arr1, arr2 in zip(ld_stats_sep[i]["sums"], ld_stats_all[i]["sums"]):
        assert np.allclose(arr1, arr2)

# parse a chromosome without the bed, and check that the results are *different*
ld_stats_no_bed = moments.LD.Parsing.compute_ld_statistics(
    "chr1.vcf.gz",
    rec_map_file="rec_map.txt",
    r_bins=r_bins,
    report=False,
    use_h5=True,
)

for arr1, arr2 in zip(ld_stats_sep[i]["sums"], ld_stats_no_bed["sums"]):
    assert not np.allclose(arr1, arr2)

# if you got here, it all worked
print("Success!")

print("cleaning up")
os.system("rm *.vcf.gz *.h5")
