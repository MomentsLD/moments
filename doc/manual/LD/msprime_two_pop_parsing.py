import msprime
import moments.LD as mold
import numpy as np
import os, sys

## First we'll create some data using msprime
## The model will be a two population model that split some time
## in the past, with subsequent migration and size changes

L = 1e6  # simulate a 1 Mb chunk
r = 2e-8  # constant per base recombination rate
u = 2e-8  # per base mutation rate

Ne = 1e4  # effective population size (ancestral size)
nu1 = 0.5  # relative size of population 1
nu2 = 2  # relative size of population 2
T = 0.2  # time of split in past (in units of 2Ne gens)
M = 1.0  # symmetric migration rate (scaled by 2Ne)

# convert to physical units
gens = int(T * 2 * Ne)
m = M / 2.0 / Ne

ns = [40, 20]  # 20 and 10 diploids in the two populations, resp.

# set up msprime simulation
population_configurations = [
    msprime.PopulationConfiguration(sample_size=ns[0], initial_size=nu1 * Ne),
    msprime.PopulationConfiguration(sample_size=ns[1], initial_size=nu2 * Ne),
]

migration_matrix = [[0, m], [m, 0]]

demographic_events = [
    msprime.MassMigration(time=gens, source=1, destination=0, proportion=1),
    msprime.MigrationRateChange(time=gens, rate=0),
    msprime.PopulationParametersChange(time=gens, initial_size=Ne, population_id=0),
]

ts = msprime.simulate(
    Ne=Ne,
    length=L,
    recombination_rate=r,
    mutation_rate=u,
    population_configurations=population_configurations,
    migration_matrix=migration_matrix,
    demographic_events=demographic_events,
)

# write the vcf
# !!! we will save simulated data in a subdirectory named 'data'
# create it if it does not already exist
with open(os.path.join("data", "two_pop.vcf"), "w+") as vcf_file:
    ts.write_vcf(vcf_file, ploidy=2)

## Create example recombination map and population file
r0 = 0
rL = r * L
with open(os.path.join("data", "rec_map.txt"), "w+") as rm_file:
    rm_file.write("pos\tmap\n")
    rm_file.write("0\t%f\n" % (r0))
    rm_file.write("%s\t%f\n" % (int(L), rL))

## default individual names in the msprime vcf output are msp_i
with open(os.path.join("data", "samples.txt"), "w+") as pop_file:
    pop_file.write("sample\tpop\n")
    for ii in range(int(sum(ns) / 2)):
        if ii < int(ns[0] / 2):
            pop_file.write("msp_{0}\tpop1\n".format(ii))
        else:
            pop_file.write("msp_{0}\tpop2\n".format(ii))


## At this point, we have all the simulated data saved

## Parse the data using moments.LD

ld_stats = mold.Parsing.compute_ld_statistics(
    "data/two_pop.vcf",
    rec_map_file="data/rec_map.txt",
    map_name="map",
    pop_file="data/samples.txt",
    pops=["pop1", "pop2"],
    cM=False,
    r_bins=[0, 1e-5, 1e-4, 1e-3],
)
