"""
Simple example of extracting a frequency spectrum from a SNP data file.
"""
import dadi

# Parse the data file to generate the data dictionary
dd = dadi.Misc.make_data_dict('data.txt')

# Extract the spectrum for ['YRI','CEU'] from that dictionary, with both
# projected down to 20 samples per population.
fs = dadi.Spectrum.from_data_dict(dd, ['YRI','CEU'], [20,20])

# Let's plot the fs.
import pylab
dadi.Plotting.plot_single_2d_sfs(fs, vmin=0.1)
pylab.show()

# If we didn't have outgroup information, we could use the folded  version 
# of the fs.
folded = dadi.Spectrum.from_data_dict(dd, ['YRI','CEU'], [20,20],
                                      polarized=False)

# We may also want to apply a statistical correction for ancestral state
# misidentification.
# To do so, we need a trinucleotide transition rate matrix. This one was
# inferred along the human lineage by Hwang and Green, PNAS 101:13994 (2004).
Q = dadi.Numerics.array_from_file('Q.HwangGreen.human.dat')
# We also need a table of trinucleotide frequencies. These are derived from
# the EGP data.
tri_freq = dict((line.split()[0], float(line.split()[1])) 
                for line in file('tri_freq.dat').readlines())
# We combine these to make a table of 1-f_{ux}, in the notation of
# of Hernandez, Williamson & Bustamante, Mol Biol Evol 24:1792 (2007).
dadi.Misc.make_fux_table('fux_table.dat', 0.0112, Q, tri_freq)
# And finally we get the corrected frequency spectrum.
fs_corr = dadi.Spectrum.from_data_dict_corrected(dd, ['YRI','CEU'], [20,20],
                                                 'fux_table.dat')
