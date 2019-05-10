imported_h5py = 0
imported_allel = 0
try:
    import allel
    imported_allel = 1
except ImportError:
    pass

try:
    import h5py
    imported_h5py = 1
except ImportError:
    pass

from . import Util

def check_imports():
    if imported_allel == 0:
        raise("Failed to import allel package needed for Parsing. Is it installed?")
    if imported_h5py == 0:
        raise("Failed to import h5py package needed for Parsing. Is it installed?")

# later go through and trim the unneeded ones
import numpy as np
import pandas
from collections import Counter
from . import stats_from_genotype_counts as sgc
from . import stats_from_haplotype_counts as shc
import sys

### does this handle only a single chromosome at a time???

def load_h5(vcf_file, report=True):
    check_imports()
    ## open the h5 callset, create if doesn't exist
    ## note that if the h5 file exists, but isn't properly written, you will need to delete and recreate
    ## saves h5 callset as same name and path, but with h5 extension instead of vcf or vcf.gz
    h5_file_path = vcf_file.split('.vcf')[0] + '.h5' # kinda hacky, sure
    try:
        callset = h5py.File(h5_file_path, mode='r')
    except (OSError,IOError): # IOError merged into OSError in python 3
        if report is True: print("creating and saving h5 file"); sys.stdout.flush()
        allel.vcf_to_hdf5(vcf_file, h5_file_path, 
                fields='*', exclude_fields=['calldata/GQ'],
                overwrite=True)
        callset = h5py.File(h5_file_path, mode='r')
    return callset


### genotype function
def get_genotypes(vcf_file, bed_file=None, chromosome=None, min_bp=None, use_h5=True, report=True):
    """
    Given a vcf file, we extract the biallelic SNP genotypes.
    If bed_file is None, we use all valid variants. Otherwise we filter genotypes
        by the given bed file.
    If chromosome (int) is given, filters to keep snps only in given chrom (useful for vcfs spanning 
        multiple chromosomes).
    min_bp : only used with bed file, filters out features that are smaller than min_bp
    If use_h5 is True, we try to load the h5 file, which has the same path/name as 
        vcf_file, but with *.h5 instead of *.vcf or *.vcf.gz. If the h5 file does not
        exist, we create it and save it as *.h5
    report : prints progress updates if True, silent otherwise
    """
    
    check_imports()
    
    if use_h5 is True:
        callset = load_h5(vcf_file, report=report)
    else:
        ## read the vcf directly
        raise ValueError("Use hdf5 format.")
    
    all_genotypes = allel.GenotypeChunkedArray(callset['calldata/GT'])
    all_positions = callset['variants/POS'][:]
    
    if report is True: print("loaded genotypes"); sys.stdout.flush()
    
    # filter SNPs not in bed file, if one is given
    if bed_file is not None: # filter genotypes and positions
        mask_bed = pandas.read_csv(bed_file, sep='\t', header=None)
        chroms = [c for c in list(set(callset['variants/CHROM'][:]))]
        
        # because of the variation of chrom labels (with our without chr (22 vs chr22)),
        # we check that chroms start with chr
        
        for ii,c in enumerate(chroms):
            if 'chr' in c:
                chroms[ii] = c[3:]
        
        chrom_filter = [False] * len(mask_bed)
        if chromosome is not None:
            chrom_filter = np.logical_or(chrom_filter, np.logical_or(mask_bed[0].values.astype(str) == str(chromosome), mask_bed[0].values.astype(str) == 'chr'+str(chromosome)))
        else:
            for chrom in chroms:
                chrom_filter = np.logical_or(chrom_filter, np.logical_or(mask_bed[0].values.astype(str) == chrom, mask_bed[0].values.astype(str) == 'chr'+chrom))
        
        mask_bed = mask_bed.loc[chrom_filter]

        # if we want a minimum length of feature, only keep long enough features
        if min_bp is not None:
            mask_bed = mask_bed.loc[mask_bed[2] - mask_bed[1] >= min_bp]
        
        n_features = mask_bed.shape[0]

        in_mask = (all_positions < 0)
        for _index, feature in mask_bed.iterrows():
            start = feature[1]
            end = feature[2]
              
            in_mask = np.logical_or(in_mask, np.logical_and(all_positions>=start, 
                                                            all_positions<end))
        if report is True: print("created bed filter"); sys.stdout.flush()
        
        all_positions = all_positions.compress(in_mask)
        all_genotypes = all_genotypes.compress(in_mask)
        
        if report is True: print("filtered by bed"); sys.stdout.flush()
    elif chromosome is not None:
        # only keep variants that are in the given chromosome number
        all_chromosomes = callset['variants/CHROM'][:]
        in_chromosome = (all_chromosomes == str(chromosome))
        all_positions = all_positions.compress(in_chromosome)
        all_genotypes = all_genotypes.compress(in_chromosome)
        
    all_genotypes_012 = all_genotypes.to_n_alt(fill=-1)
    
    # count alleles and only keep biallelic positions
    allele_counts = all_genotypes.count_alleles()
    is_biallelic = allele_counts.is_biallelic_01()
    biallelic_positions = all_positions.compress(is_biallelic)
    
    biallelic_genotypes_012 = all_genotypes_012.compress(is_biallelic)
    
    biallelic_allele_counts = allele_counts.compress(is_biallelic)
    biallelic_genotypes = all_genotypes.compress(is_biallelic)
    
    if report is True: print("kept biallelic positions"); sys.stdout.flush()
    
    relevant_column = np.array([False] * biallelic_allele_counts.shape[1])
    relevant_column[0:2] = True
    biallelic_allele_counts = biallelic_allele_counts.compress(relevant_column, axis = 1)
    
    sample_ids = callset['samples']
    
    return biallelic_positions, biallelic_genotypes, biallelic_allele_counts, sample_ids


def assign_r_pos(positions, rec_map):
    rs = np.zeros(len(positions))
    for ii,pos in enumerate(positions):
        if pos in np.array(rec_map[0]):
            rs[ii] = np.array(rec_map[1])[np.argwhere(pos == np.array(rec_map[0]))[0]] 
        else:
            ## for now, if outside rec map, assign to nearest point, but later want to drop these positions
            if pos < rec_map[0].iloc[0]:
                rs[ii] = rec_map[1].iloc[0]
            elif pos > rec_map[0].iloc[-1]:
                rs[ii] = rec_map[1].iloc[-1]
            else:
                map_ii = np.where(pos >= np.array(rec_map[0]))[0][-1]
                l = rec_map[0][map_ii]
                r = rec_map[0][map_ii+1]
                v_l = rec_map[1][map_ii]
                v_r = rec_map[1][map_ii+1]
                rs[ii] = v_l + (v_r-v_l) * (pos-l)/(r-l)
    return rs


def assign_recombination_rates(positions, map_file, map_name=None, map_sep='\t', cM=True, report=True):
    if map_file == None:
        raise ValueError("Need to pass a recombination map file. Otherwise can bin by physical distance."); sys.stdout.flush()
    try:
        rec_map = pandas.read_csv(map_file, sep=map_sep)
    except:
        raise ValueError("Error loading map."); sys.stdout.flush()
    
    if map_name == None: # we use the first map column
        print("No recombination map name given, using first column."); sys.stdout.flush()
    else:
        map_positions = rec_map[rec_map.keys()[0]]
        try:
            map_values = rec_map[map_name]
        except KeyError:
            print("WARNING: map_name did not match map names in recombination map file. Using first column...")
            map_values = rec_map[rec_map.keys()[1]]
        
    # for positions sticking out the end of the map, they take the value of the closest position
    # ideally, you'd filter these out
    
    if cM == True:
        map_values /= 100
    
    pos_rs = assign_r_pos(positions, [map_positions, map_values])
    
    return pos_rs

"""
For counting genopytes and computing D^2, Dz, and pi2 for a block of genotypes
vvv
"""

def count_genotypes(Gs):
    """
    Gs is a matrix, where rows correspond to loci (number of loci = L), and
    the columns correspond to individuals (sample size = n)
    so Gs has size L x n
    
    There are L*(L-1)/2 total comparisons to make
    """
    L,n = np.shape(Gs)
    
    X = np.empty((L*(L-1)//2, n)).astype('int')
    c = 0
    for i in range(L)[:-1]:
        X[c:c+L-i-1] = 3*Gs[i] + Gs[i+1:]
        c += (L-i-1)
    
    Counts = np.empty((L*(L-1)//2, 9)).astype('int')
    for i in range(len(Counts)):
        Counts[i] = np.bincount(X[i], minlength=9)
    
    return np.fliplr(Counts)

def count_genotypes_between(Gs1, Gs2):
    """
    The Gs are matrices, where rows correspond to loci and columns to individuals
    Both matrices must have the same number of individuals
    If Gs1 has length L1 and Gs2 has length L2, we compute all pairwise counts,
    return Counts, which has size (L1*L2, 9) 
    """
    L1,n1 = np.shape(Gs1)
    L2,n2 = np.shape(Gs2)
    
    if n1 != n2:
        raise ValueError("data must have same number of sequenced individuals")
    else:
        n = n1
    
    X = np.empty((L1*L2, n)).astype('int')
    for i in range(L1):
        X[i*L2:(i+1)*L2] = 3*Gs1[i] + Gs2
    
    Counts = np.empty((L1*L2, 9)).astype('int')
    for i in range(len(Counts)):
        Counts[i] = np.bincount(X[i], minlength=9)
    
    return np.fliplr(Counts)

def compute_D2(Counts):
    """
    D2 for block of genotypes counted from count_genotypes(Gs)
    """
    n1 = Counts[:,0]
    n2 = Counts[:,1]
    n3 = Counts[:,2]
    n4 = Counts[:,3]
    n5 = Counts[:,4]
    n6 = Counts[:,5]
    n7 = Counts[:,6]
    n8 = Counts[:,7]
    n9 = Counts[:,8]
    n = np.sum(Counts, axis=1)
    numer = (n2*n4 - n2**2*n4 + 4*n3*n4 - 4*n2*n3*n4 - 4*n3**2*n4 - n2*n4**2 - 4*n3*n4**2 + n1*n5 - n1**2*n5 + n3*n5 + 2*n1*n3*n5 - n3**2*n5 - 4*n3*n4*n5 - n1*n5**2 - n3*n5**2 + 4*n1*n6 - 4*n1**2*n6 + n2*n6 - 4*n1*n2*n6 - n2**2*n6 + 2*n2*n4*n6 - 4*n1*n5*n6 - 4*n1*n6**2 - n2*n6**2 + 4*n2*n7 - 4*n2**2*n7 + 16*n3*n7 - 16*n2*n3*n7 - 16*n3**2*n7 - 4*n2*n4*n7 - 16*n3*n4*n7 + n5*n7 + 2*n1*n5*n7 - 4*n2*n5*n7 - 18*n3*n5*n7 - n5**2*n7 + 4*n6*n7 + 8*n1*n6*n7 - 16*n3*n6*n7 - 4*n5*n6*n7 - 4*n6**2*n7 - 4*n2*n7**2 - 16*n3*n7**2 - n5*n7**2 - 4*n6*n7**2 + 4*n1*n8 - 4*n1**2*n8 + 4*n3*n8 + 8*n1*n3*n8 - 4*n3**2*n8 + n4*n8 - 4*n1*n4*n8 + 2*n2*n4*n8 - n4**2*n8 - 4*n1*n5*n8 - 4*n3*n5*n8 + n6*n8 + 2*n2*n6*n8 - 4*n3*n6*n8 + 2*n4*n6*n8 - n6**2*n8 - 16*n3*n7*n8 - 4*n6*n7*n8 - 4*n1*n8**2 - 4*n3*n8**2 - n4*n8**2 - n6*n8**2 + 16*n1*n9 - 16*n1**2*n9 + 4*n2*n9 - 16*n1*n2*n9 - 4*n2**2*n9 + 4*n4*n9 - 16*n1*n4*n9 + 8*n3*n4*n9 - 4*n4**2*n9 + n5*n9 - 18*n1*n5*n9 - 4*n2*n5*n9 + 2*n3*n5*n9 - 4*n4*n5*n9 - n5**2*n9 - 16*n1*n6*n9 - 4*n2*n6*n9 + 8*n2*n7*n9 + 2*n5*n7*n9 - 16*n1*n8*n9 - 4*n4*n8*n9 - 16*n1*n9**2 - 4*n2*n9**2 - 4*n4*n9**2 - n5*n9**2)/16. + (-((n2/2. + n3 + n5/4. + n6/2.)*(n4/2. + n5/4. + n7 + n8/2.)) + (n1 + n2/2. + n4/2. + n5/4.)*(n5/4. + n6/2. + n8/2. + n9))**2
    denom = 1.*n*(n-1)*(n-2)*(n-3)
    return 4. * numer / denom     ### check factor of four


def compute_Dz(Counts):
    """
    Dz for block of genotypes counted from count_genotypes(Gs)
    """
    n1 = Counts[:,0]
    n2 = Counts[:,1]
    n3 = Counts[:,2]
    n4 = Counts[:,3]
    n5 = Counts[:,4]
    n6 = Counts[:,5]
    n7 = Counts[:,6]
    n8 = Counts[:,7]
    n9 = Counts[:,8]
    n = np.sum(Counts, axis=1)
    numer = (-(n2*n4) + 3*n1*n2*n4 + n2**2*n4 + 2*n3*n4 + 4*n1*n3*n4 - n2*n3*n4 - 4*n3**2*n4 + n2*n4**2 + 2*n3*n4**2 + 2*n1*n5 - 3*n1**2*n5 - n1*n2*n5 + 2*n3*n5 + 2*n1*n3*n5 - n2*n3*n5 - 3*n3**2*n5 - n1*n4*n5 + n3*n4*n5 + 2*n1*n6 - 4*n1**2*n6 - n2*n6 - n1*n2*n6 + n2**2*n6 + 4*n1*n3*n6 + 3*n2*n3*n6 - 2*n1*n4*n6 - 2*n2*n4*n6 - 2*n3*n4*n6 + n1*n5*n6 - n3*n5*n6 + 2*n1*n6**2 + n2*n6**2 + 2*n2*n7 + 4*n1*n2*n7 + 2*n2**2*n7 + 8*n3*n7 + 4*n1*n3*n7 - 4*n3**2*n7 - n2*n4*n7 + 2*n5*n7 + 2*n1*n5*n7 + n2*n5*n7 + 2*n3*n5*n7 - n4*n5*n7 + 2*n6*n7 - n2*n6*n7 - 2*n4*n6*n7 + n5*n6*n7 + 2*n6**2*n7 - 4*n2*n7**2 - 4*n3*n7**2 - 3*n5*n7**2 - 4*n6*n7**2 + 2*n1*n8 - 4*n1**2*n8 - 2*n1*n2*n8 + 2*n3*n8 - 2*n2*n3*n8 - 4*n3**2*n8 - n4*n8 - n1*n4*n8 - 2*n2*n4*n8 - n3*n4*n8 + n4**2*n8 + n1*n5*n8 + n3*n5*n8 - n6*n8 - n1*n6*n8 - 2*n2*n6*n8 - n3*n6*n8 - 2*n4*n6*n8 + n6**2*n8 + 4*n1*n7*n8 - 2*n2*n7*n8 + 3*n4*n7*n8 - n5*n7*n8 - n6*n7*n8 + 2*n1*n8**2 + 2*n3*n8**2 + n4*n8**2 + n6*n8**2 + 8*n1*n9 - 4*n1**2*n9 + 2*n2*n9 + 2*n2**2*n9 + 4*n1*n3*n9 + 4*n2*n3*n9 + 2*n4*n9 - n2*n4*n9 + 2*n4**2*n9 + 2*n5*n9 + 2*n1*n5*n9 + n2*n5*n9 + 2*n3*n5*n9 + n4*n5*n9 - n2*n6*n9 - 2*n4*n6*n9 - n5*n6*n9 + 4*n1*n7*n9 + 4*n3*n7*n9 + 4*n4*n7*n9 + 2*n5*n7*n9 + 4*n6*n7*n9 - 2*n2*n8*n9 + 4*n3*n8*n9 - n4*n8*n9 - n5*n8*n9 + 3*n6*n8*n9 - 4*n1*n9**2 - 4*n2*n9**2 - 4*n4*n9**2 - 3*n5*n9**2)/4. + (-n1 + n3 - n4 + n6 - n7 + n9)*(-n1 - n2 - n3 + n7 + n8 + n9)*(-((n2/2. + n3 + n5/4. + n6/2.)*(n4/2. + n5/4. + n7 + n8/2.)) + (n1 + n2/2. + n4/2. + n5/4.)*(n5/4. + n6/2. + n8/2. + n9))
    denom = 1.*n*(n-1)*(n-2)*(n-3) 
    return 2.*numer / denom  ### check factor


def compute_pi2(Counts):
    """
    pi2 for block of genotypes counted from count_genotypes(Gs)
    """
    n1 = Counts[:,0]
    n2 = Counts[:,1]
    n3 = Counts[:,2]
    n4 = Counts[:,3]
    n5 = Counts[:,4]
    n6 = Counts[:,5]
    n7 = Counts[:,6]
    n8 = Counts[:,7]
    n9 = Counts[:,8]
    n = np.sum(Counts, axis=1)
    numer = (n1 + n2 + n3 + n4/2. + n5/2. + n6/2.)*(n1 + n2/2. + n4 + n5/2. + n7 + n8/2.)*(n2/2. + n3 + n5/2. + n6 + n8/2. + n9)*(n4/2. + n5/2. + n6/2. + n7 + n8 + n9) + (13*n2*n4 - 16*n1*n2*n4 - 11*n2**2*n4 + 16*n3*n4 - 28*n1*n3*n4 - 24*n2*n3*n4 - 8*n3**2*n4 - 11*n2*n4**2 - 20*n3*n4**2 - 6*n5 + 12*n1*n5 - 4*n1**2*n5 + 17*n2*n5 - 20*n1*n2*n5 - 11*n2**2*n5 + 12*n3*n5 - 28*n1*n3*n5 - 20*n2*n3*n5 - 4*n3**2*n5 + 17*n4*n5 - 20*n1*n4*n5 - 32*n2*n4*n5 - 40*n3*n4*n5 - 11*n4**2*n5 + 11*n5**2 - 16*n1*n5**2 - 17*n2*n5**2 - 16*n3*n5**2 - 17*n4*n5**2 - 6*n5**3 + 16*n1*n6 - 8*n1**2*n6 + 13*n2*n6 - 24*n1*n2*n6 - 11*n2**2*n6 - 28*n1*n3*n6 - 16*n2*n3*n6 + 24*n4*n6 - 36*n1*n4*n6 - 38*n2*n4*n6 - 36*n3*n4*n6 - 20*n4**2*n6 + 17*n5*n6 - 40*n1*n5*n6 - 32*n2*n5*n6 - 20*n3*n5*n6 - 42*n4*n5*n6 - 17*n5**2*n6 - 20*n1*n6**2 - 11*n2*n6**2 - 20*n4*n6**2 - 11*n5*n6**2 + 16*n2*n7 - 28*n1*n2*n7 - 20*n2**2*n7 + 16*n3*n7 - 48*n1*n3*n7 - 44*n2*n3*n7 - 16*n3**2*n7 - 24*n2*n4*n7 - 44*n3*n4*n7 + 12*n5*n7 - 28*n1*n5*n7 - 40*n2*n5*n7 - 48*n3*n5*n7 - 20*n4*n5*n7 - 16*n5**2*n7 + 16*n6*n7 - 48*n1*n6*n7 - 48*n2*n6*n7 - 44*n3*n6*n7 - 36*n4*n6*n7 - 40*n5*n6*n7 - 20*n6**2*n7 - 8*n2*n7**2 - 16*n3*n7**2 - 4*n5*n7**2 - 8*n6*n7**2 + 16*n1*n8 - 8*n1**2*n8 + 24*n2*n8 - 36*n1*n2*n8 - 20*n2**2*n8 + 16*n3*n8 - 48*n1*n3*n8 - 36*n2*n3*n8 - 8*n3**2*n8 + 13*n4*n8 - 24*n1*n4*n8 - 38*n2*n4*n8 - 48*n3*n4*n8 - 11*n4**2*n8 + 17*n5*n8 - 40*n1*n5*n8 - 42*n2*n5*n8 - 40*n3*n5*n8 - 32*n4*n5*n8 - 17*n5**2*n8 + 13*n6*n8 - 48*n1*n6*n8 - 38*n2*n6*n8 - 24*n3*n6*n8 - 38*n4*n6*n8 - 32*n5*n6*n8 - 11*n6**2*n8 - 28*n1*n7*n8 - 36*n2*n7*n8 - 44*n3*n7*n8 - 16*n4*n7*n8 - 20*n5*n7*n8 - 24*n6*n7*n8 - 20*n1*n8**2 - 20*n2*n8**2 - 20*n3*n8**2 - 11*n4*n8**2 - 11*n5*n8**2 - 11*n6*n8**2 + 16*n1*n9 - 16*n1**2*n9 + 16*n2*n9 - 44*n1*n2*n9 - 20*n2**2*n9 - 48*n1*n3*n9 - 28*n2*n3*n9 + 16*n4*n9 - 44*n1*n4*n9 - 48*n2*n4*n9 - 48*n3*n4*n9 - 20*n4**2*n9 + 12*n5*n9 - 48*n1*n5*n9 - 40*n2*n5*n9 - 28*n3*n5*n9 - 40*n4*n5*n9 - 16*n5**2*n9 - 44*n1*n6*n9 - 24*n2*n6*n9 - 36*n4*n6*n9 - 20*n5*n6*n9 - 48*n1*n7*n9 - 48*n2*n7*n9 - 48*n3*n7*n9 - 28*n4*n7*n9 - 28*n5*n7*n9 - 28*n6*n7*n9 - 44*n1*n8*n9 - 36*n2*n8*n9 - 28*n3*n8*n9 - 24*n4*n8*n9 - 20*n5*n8*n9 - 16*n6*n8*n9 - 16*n1*n9**2 - 8*n2*n9**2 - 8*n4*n9**2 - 4*n5*n9**2)/16.
    denom = 1.*n*(n-1)*(n-2)*(n-3)
    return numer / denom

def compute_pairwise_stats(Gs):
    Counts = count_genotypes(Gs)
    D2 = compute_D2(Counts)
    Dz = compute_Dz(Counts)
    pi2 = compute_pi2(Counts)
    return D2, Dz, pi2

def compute_average_stats(Gs):
    D2, Dz, pi2 = compute_pairwise_stats(Gs)
    return np.mean(D2), np.mean(Dz), np.mean(pi2)

def compute_pairwise_stats_between(Gs1,Gs2):
    Counts = count_genotypes_between(Gs1,Gs2)
    D2 = compute_D2(Counts)
    Dz = compute_Dz(Counts)
    pi2 = compute_pi2(Counts)
    return D2, Dz, pi2

def compute_average_stats_between(Gs1,Gs2):
    D2, Dz, pi2 = compute_pairwise_stats_between(Gs1,Gs2)
    return np.mean(D2), np.mean(Dz), np.mean(pi2)


"""
^^^
"""


def g_tally_counter(g_l, g_r):
    #### Counter on iterable
    #gs = list(zip(g_l,g_r))
    c = Counter(zip(g_l,g_r))
    return (c[(2,2)], c[(2,1)], c[(2,0)], 
            c[(1,2)], c[(1,1)], c[(1,0)], 
            c[(0,2)], c[(0,1)], c[(0,0)])

def g_tally_counter_2(g_l,g_r):
    return tuple(np.bincount(3*g_l + g_r, minlength=9)[::-1]) ### could save more time by figuring out how not to switch between tuples and lists and arrays all the time...

def g_tally_counter_3(g_l, g_r):
    gg = g_l*3 + g_r
    m = gg.shape[0]
    n = 9
    A1 = (gg.T + (n*np.arange(m))).T
    out = np.bincount(A1.ravel(),minlength=n*m).reshape(m,-1)
    return out[:,::-1]

def h_tally_counter(h_l, h_r):
    #### Counter on iterable
    hs = list(zip(h_l, h_r))
    c = Counter(hs)
    return (c[(1,1)], c[(1,0)], c[(0,1)], c[(0,0)])

def h_tally_counter_3(h_l, h_r):
    hh = h_l*2 + h_r
    m = hh.shape[0]
    n = 4
    A1 = (hh.T + (4*np.arange(m))).T
    out = np.bincount(A1.ravel(),minlength=n*m).reshape(m,-1)
    return out[:,::-1]


def count_types(genotypes, bins, sample_ids, positions=None, pos_rs=None, pop_file=None, pops=None, use_genotypes=True, report=True, report_spacing=1000, use_cache=True, stats_to_compute=None):
    """
    genotypes : in format of 0,1,2
    bins : bin edges, either recombination distances or bp distances
    sample_ids : 
    positions : 
    pos_rs : 
    pop_file : 
    pops : 
    use_genotypes : if true, we use genotype values 012, if False, we assume genotypes are phased
    report : prints updates if True
    report_spacing : how often to report (if True) as we parse through positions
    use_cache : if True, we cache count types over each bin and later compute statistics. The caches
                could require too much memory, in which case we compute statistics as we count pairs
    stats_to_compute : passed only if use_cache = False
    """
    
    if report==True: print("keeping only variable positions in these pops"); sys.stdout.flush()
    pop_indexes = {}
    if pops is not None:
        samples = pandas.read_csv(pop_file, sep='\t')
        cols_to_keep = np.array([False]*np.shape(genotypes)[1])
        all_samples_to_keep = []
        for pop in pops:
            all_samples_to_keep += list(samples[samples['pop'] == pop]['sample'])
        
        
        for s in all_samples_to_keep:
            cols_to_keep[list(sample_ids).index(s)] = True
        
        # keep only biallelic genotypes from populations in pops, discard the rest
        genotypes_pops = genotypes.compress(cols_to_keep, axis=1)
        allele_counts_pops = genotypes_pops.count_alleles()
        is_biallelic = allele_counts_pops.is_biallelic_01()
        genotypes_pops = genotypes_pops.compress(is_biallelic)
        
        sample_ids_pops = list(np.array(list(samples['sample'])).compress(cols_to_keep))
        
        for pop in pops:
            pop_indexes[pop] = np.array([False]*np.shape(genotypes_pops)[1])
            for s in samples[samples['pop'] == pop]['sample']:
                pop_indexes[pop][sample_ids_pops.index(s)] = True
        
        if use_genotypes == False:
            pop_indexes_haps = {}
            for pop in pops:
                pop_indexes_haps[pop] = np.reshape(list(zip(pop_indexes[pop], pop_indexes[pop])),(2*len(pop_indexes[pop]),))
        
        if positions is not None:
            positions = positions.compress(is_biallelic)
        if pos_rs is not None:
            pos_rs = pos_rs.compress(is_biallelic)
        
    else:
        print("No populations given, using all samples as one population."); sys.stdout.flush()
        pops = ['ALL']
        pop_indexes['ALL'] = np.array([True]*np.shape(genotypes)[1])
        genotypes_pops = genotypes
        if use_genotypes == False:
            pop_indexes_haps = {}
            for pop in pops:
                pop_indexes_haps[pop] = np.reshape(list(zip(pop_indexes[pop], pop_indexes[pop])),(2*len(pop_indexes[pop]),))

    
    if use_genotypes == True:
        genotypes_pops_012 = genotypes_pops.to_n_alt()
    else:
        haplotypes_pops_01 = genotypes_pops.to_haplotypes()
    
    ## only keep biallelic positions that are variable in the populations we care about
    
    bs = list(zip(bins[:-1],bins[1:]))
    
    ## type_counts will store the number of times we see each genotype count configuration, within each bin
    if use_cache == True:
        type_counts = {}
        for b in bs:
            type_counts[b] = {}
    else:
        sums = {}
        for b in bs:
            sums[b] = {}
            for stat in stats_to_compute[0]:
                sums[b][stat] = 0
        
    if pos_rs is not None:
        rs = pos_rs
    elif positions is not None:
        rs = positions
    
    ns = np.array([2 * sum(pop_indexes[pop]) for pop in pops])
    
    bins = np.array(bins)
    
    ## here, we split our genotype array into sub-genotype arrays for each population
    if use_cache == True: # if we use caches, it's faster to look up genotype arrays
        if report==True: print("creating look-up dict for genotypes"); sys.stdout.flush()
        if use_genotypes == True:        
            genotypes_by_pop = {}
            for pop in pops:
                #genotypes_by_pop[pop] = genotypes_pops_012.compress(pop_indexes[pop], axis=1)
                temp_genotypes = genotypes_pops_012.compress(pop_indexes[pop], axis=1)
                genotypes_by_pop[pop] = {}
                for ii in range(len(temp_genotypes)):
                    genotypes_by_pop[pop][ii] = temp_genotypes[ii]
        else:
            haplotypes_by_pop = {}
            for pop in pops:
                #haplotypes_by_pop[pop] = haplotypes_pops_01.compress(pop_indexes_haps[pop], axis=1)
                temp_haplotypes = haplotypes_pops_01.compress(pop_indexes_haps[pop], axis=1)
                haplotypes_by_pop[pop] = {}
                for ii in range(len(temp_haplotypes)):
                    haplotypes_by_pop[pop][ii] = temp_haplotypes[ii]
    else: # don't want to use dict caches for genotype arrays, so we have to read from the genotype arrays each time
        if report==True: print("pre-compressing for variant loci"); sys.stdout.flush()
        if use_genotypes == True:        
            genotypes_by_pop = {}
            for pop in pops:
                genotypes_by_pop[pop] = genotypes_pops_012.compress(pop_indexes[pop], axis=1)
        else:
            haplotypes_by_pop = {}
            for pop in pops:
                haplotypes_by_pop[pop] = haplotypes_pops_01.compress(pop_indexes_haps[pop], axis=1)
    
    # loop through 'left' positions, paired with positions to the right
    for ii,r in enumerate(rs[:-1]):
        if report is True:
            if ii%report_spacing == 0:
                print("tallied two locus counts {0} of {1} positions".format(ii, len(rs))); sys.stdout.flush()
        
        ## extract the genotypes at the left locus just once
        if use_genotypes == True:
            genotypes_left = [genotypes_by_pop[pop][ii] for pop in pops]
        else:
            haplotypes_left = [haplotypes_by_pop[pop][ii] for pop in pops]
        
        ## loop through each bin, picking out the positions to the right of the left locus that fall within the given bin
        #for b in bs:
        r_dists = pos_rs - r
        
        #filt = np.logical_and(pos_rs - r >= b[0], pos_rs - r < b[1])
        filt = np.logical_and(pos_rs - r >= bs[0][0], pos_rs - r < bs[-1][1])
        filt[ii] = False
        right_indices = np.where(filt == True)[0]
        
        ## if there are no variants within the bin's distance to the right, continue to next bin 
        if len(right_indices) == 0:
            continue
        
        right_start = right_indices[0]
        right_end = right_indices[-1]+1
        
        if use_cache == True:
            # we stored genotypes in a dictionary
            if use_genotypes == True:
                genotypes_right = []
                for pop_ind,pop in enumerate(pops):
                    genotypes_right_pop = np.empty((right_end-right_start, int(ns[pop_ind]/2))).astype('int')
                    for right_ind in range(right_start,right_end):
                        genotypes_right_pop[right_ind-right_start] = genotypes_by_pop[pop][right_ind]
                    genotypes_right.append(genotypes_right_pop)
            else:
                haplotypes_right = []
                for pop_ind,pop in enumerate(pops):
                    haplotypes_right_pop = np.empty((right_end-right_start, ns[pop_ind])).astype('int')
                    for right_ind in range(right_start,right_end):
                        haplotypes_right_pop[right_ind-right_start] = haplotypes_by_pop[pop][right_ind]
                    haplotypes_right.append(haplotypes_right_pop)
        else:
            # we read from a slice of the genotype arrays (which for some reason is slower)
            if use_genotypes == True:
                genotypes_right = [genotypes_by_pop[pop][right_start:right_end] for pop in pops]
            else:
                haplotypes_right = [haplotypes_by_pop[pop][right_start:right_end] for pop in pops]
        
        if use_genotypes == True:
            cs = [ g_tally_counter_3(genotypes_left[pop_ind], genotypes_right[pop_ind]) for pop_ind in range(len(pops)) ]
        else:
            cs = [ h_tally_counter_3(haplotypes_left[pop_ind], haplotypes_right[pop_ind]) for pop_ind in range(len(pops)) ]
                
        for jj,r_pos in enumerate(r_dists[right_start:right_end]):
            bin_ind = np.where(r_pos >= bins)[0][-1]
            b = bs[bin_ind]
            
            cs_ind = tuple([tuple(cs[pop_ind][jj]) for pop_ind in range(len(pops))])
            
            if use_cache == True:
                type_counts[b].setdefault(cs_ind,0)
                type_counts[b][cs_ind] += 1
            else:
                for stat in stats_to_compute[0]:
                    sums[b][stat] += call_sgc(stat, cs_ind, use_genotypes)
                
    if use_cache == True:
        return type_counts
    else:
        return sums


def call_sgc(stat, Cs, use_genotypes):
    s = stat.split('_')[0]
    pop_nums = [int(p)-1 for p in stat.split('_')[1:]]
    if s == 'DD':
        if use_genotypes == True:
            return sgc.DD(Cs, pop_nums)
        else:
            return shc.DD(Cs, pop_nums)
    if s == 'Dz':
        ii,jj,kk = pop_nums
        if jj == kk:
            if use_genotypes == True:
                return sgc.Dz(Cs, pop_nums)
            else:
                return shc.Dz(Cs, pop_nums)
        else:
            if use_genotypes == True:
                return 1./2 * sgc.Dz(Cs, [ii,jj,kk]) + 1./2 * sgc.Dz(Cs, [ii,kk,jj])
            else:
                return 1./2 * shc.Dz(Cs, [ii,jj,kk]) + 1./2 * shc.Dz(Cs, [ii,kk,jj])
    if s == 'pi2':
        ii,jj,kk,ll = pop_nums ### this doesn't consider the symmetry between p/q yet...
        if ii == jj:
            if kk == ll:
                if ii == kk: # all the same
                    if use_genotypes == True:
                        return sgc.pi2(Cs, [ii,jj,kk,ll])
                    else:
                        return shc.pi2(Cs, [ii,jj,kk,ll])
                else: # (1, 1; 2, 2)
                    if use_genotypes == True:
                        return 1./2 * (sgc.pi2(Cs, [ii,jj,kk,ll]) + sgc.pi2(Cs, [kk,ll,ii,jj]) )
                    else:
                        return 1./2 * (shc.pi2(Cs, [ii,jj,kk,ll]) + shc.pi2(Cs, [kk,ll,ii,jj]) )
            else: # (1, 1; 2, 3) or (1, 1; 1, 2)
                if use_genotypes == True:
                    return 1./4 * ( sgc.pi2(Cs, [ii,jj,kk,ll]) + sgc.pi2(Cs, [ii,jj,ll,kk]) + sgc.pi2(Cs, [kk,ll,ii,jj]) + sgc.pi2(Cs, [ll,kk,ii,jj]) )
                else:
                    return 1./4 * ( shc.pi2(Cs, [ii,jj,kk,ll]) + shc.pi2(Cs, [ii,jj,ll,kk]) + shc.pi2(Cs, [kk,ll,ii,jj]) + shc.pi2(Cs, [ll,kk,ii,jj]) )
        else:
            if kk == ll: # (1, 2; 3, 3) or (1, 2; 2, 2)
                if use_genotypes == True:
                    return 1./4 * ( sgc.pi2(Cs, [ii,jj,kk,ll]) + sgc.pi2(Cs, [jj,ii,kk,ll]) + sgc.pi2(Cs, [kk,ll,ii,jj]) + sgc.pi2(Cs, [kk,ll,jj,ii]) )
                else:
                    return 1./4 * ( shc.pi2(Cs, [ii,jj,kk,ll]) + shc.pi2(Cs, [jj,ii,kk,ll]) + shc.pi2(Cs, [kk,ll,ii,jj]) + shc.pi2(Cs, [kk,ll,jj,ii]) )
            else: # (1, 2; 3, 4)
                if use_genotypes == True:
                    return 1./8 * ( sgc.pi2(Cs, [ii,jj,kk,ll]) + sgc.pi2(Cs, [ii,jj,ll,kk]) + sgc.pi2(Cs, [jj,ii,kk,ll]) + sgc.pi2(Cs, [jj,ii,ll,kk]) + sgc.pi2(Cs, [kk,ll,ii,jj]) + sgc.pi2(Cs, [ll,kk,ii,jj]) + sgc.pi2(Cs, [kk,ll,jj,ii]) + sgc.pi2(Cs, [ll,kk,jj,ii]) )
                else:
                    return 1./8 * ( shc.pi2(Cs, [ii,jj,kk,ll]) + shc.pi2(Cs, [ii,jj,ll,kk]) + shc.pi2(Cs, [jj,ii,kk,ll]) + shc.pi2(Cs, [jj,ii,ll,kk]) + shc.pi2(Cs, [kk,ll,ii,jj]) + shc.pi2(Cs, [ll,kk,ii,jj]) + shc.pi2(Cs, [kk,ll,jj,ii]) + shc.pi2(Cs, [ll,kk,jj,ii]) )


def cache_ld_statistics(type_counts, ld_stats, bins, use_genotypes=True, report=True):
    bs = list(zip(bins[:-1],bins[1:]))
    
    estimates = {}
    for b in bs:
        for cs in type_counts[b].keys():
            estimates.setdefault(cs, {})
    
    all_counts = np.array(list(estimates.keys()))
    all_counts = np.swapaxes(all_counts,0,1)
    all_counts = np.swapaxes(all_counts,1,2)
    
    for stat in ld_stats:
        if report is True: print("computing " + stat); sys.stdout.flush()
        vals = call_sgc(stat, all_counts, use_genotypes)
        for ii in range(len(all_counts[0,0])):
            cs = all_counts[:,:,ii]
            estimates[tuple([tuple(c) for c in cs])][stat] = vals[ii]
    return estimates


def get_H_statistics(genotypes, sample_ids, pop_file=None, pops=None):
    """
    Het values are not normalized by sequence length, would need to compute L from bed file.
    """
    
    if pop_file == None and pops == None:
        print("No population file or population names given, assuming all samples as single pop."); sys.stdout.flush()
    elif pops == None:
        raise ValueError("pop_file given, but not population names..."); sys.stdout.flush()
    elif pop_file == None:
        raise ValueError("Population names given, but not pop_file..."); sys.stdout.flush()
    
    if pops == None:
        pops = ['ALL']
    
    if pop_file is not None:
        samples = pandas.read_csv(pop_file, sep='\t')
        populations = np.array(samples['pop'].value_counts().keys())
        samples.reset_index(drop=True, inplace=True)

        ### should use this above when counting two locus genotypes

        subpops = {
            # for each population, get the list of samples that belong to the population
            pop_iter: samples[samples['pop'] == pop_iter].index.tolist() for pop_iter in pops
        }
        
        ac_subpop = genotypes.count_alleles_subpops(subpops)
    else:
        subpops = {
            pop_iter: list(range(len(sample_ids))) for pop_iter in pops
        }
        ac_subpop = genotypes.count_alleles_subpops(subpops)
    
    Hs = {}
    for ii,pop1 in enumerate(pops):
        for pop2 in pops[ii:]:
            if pop1 == pop2:
                H = np.sum( 2. * ac_subpop[pop1][:,0] * ac_subpop[pop1][:,1] / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1]) / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1] - 1) )
            else:
                H = np.sum( 
                        1. * ac_subpop[pop1][:,0] * ac_subpop[pop2][:,1] / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1]) / (ac_subpop[pop2][:,0] + ac_subpop[pop2][:,1]) 
                        + 1. * ac_subpop[pop1][:,1] * ac_subpop[pop2][:,0] / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1]) / (ac_subpop[pop2][:,0] + ac_subpop[pop2][:,1]) 
                    )
            Hs[(pop1,pop2)] = H
    
    return Hs


def get_reported_stats(genotypes, bins, sample_ids, positions=None, pos_rs=None, pop_file=None, pops=None, use_genotypes=True, report=True, report_spacing=1000, use_cache=True, stats_to_compute=None):
    ### build wrapping function that can take use_cache = True or False
    # now if bins is empty, we only return heterozygosity statistics
    
    if stats_to_compute == None:
        if pops is None:
            stats_to_compute = Util.moment_names(1)
        else:
            stats_to_compute = Util.moment_names(len(pops))
    
    bs = list(zip(bins[:-1],bins[1:]))
    
    if use_cache == True:
        type_counts = count_types(genotypes, bins, sample_ids, positions=positions, pos_rs=pos_rs, pop_file=pop_file, pops=pops, use_genotypes=use_genotypes, report=report, report_spacing=report_spacing, use_cache=use_cache)
        
        if report is True: print("counted genotypes"); sys.stdout.flush()
        statistics_cache = cache_ld_statistics(type_counts, stats_to_compute[0], bins, use_genotypes=use_genotypes, report=report)
        
        sums = {}
        for b in bs:
            sums[b] = {}
            for stat in stats_to_compute[0]:
                sums[b][stat] = 0
                for cs in type_counts[b]:
                    sums[b][stat] += type_counts[b][cs] * statistics_cache[cs][stat]
        
    else:
        sums = count_types(genotypes, bins, sample_ids, positions=positions, pos_rs=pos_rs, pop_file=pop_file, pops=pops, use_genotypes=use_genotypes, report=report, report_spacing=report_spacing, use_cache=use_cache, stats_to_compute=stats_to_compute)
    
    if report is True: print("computed sums\ngetting heterozygosity statistics"); sys.stdout.flush()
        
    if len(stats_to_compute[1]) == 0:
        Hs = {}
    else:
        Hs = get_H_statistics(genotypes, sample_ids, pop_file=pop_file, pops=pops)
    
    reported_stats = {}
    reported_stats['bins'] = bs
    reported_stats['sums'] = [np.empty(len(stats_to_compute[0])) for b in bs] + [np.empty(len(stats_to_compute[1]))]
    for ii,b in enumerate(bs):
        for s in stats_to_compute[0]:
            reported_stats['sums'][ii][stats_to_compute[0].index(s)] = sums[b][s]
    
    if pops == None:
        pops = ['ALL']
    for s in stats_to_compute[1]:
        reported_stats['sums'][-1][stats_to_compute[1].index(s)] = Hs[(pops[int(s.split('_')[1])-1],pops[int(s.split('_')[2])-1])]
    reported_stats['stats'] = stats_to_compute
    reported_stats['pops'] = pops
    return reported_stats


def compute_ld_statistics(vcf_file, bed_file=None, chromosome=None, rec_map_file=None, map_name=None, map_sep='\t', pop_file=None, pops=None, cM=True, r_bins=None, bp_bins=None, min_bp=None, use_genotypes=True, use_h5=True, stats_to_compute=None, report=True, report_spacing=1000, use_cache=True):
    """
    vcf_file : path to vcf file
    bed_file : path to bed file to specify regions over which to compute LD statistics. If None, computes statistics
               for all positions in vcf_file
    rec_map_file : path to recombination map
    map_name : if None, takes the first map column, otherwise takes the specified map column
    map_sep : tells pandas how to parse the recombination map. Default is tabs, though I've been working 
              with space delimitted map files
    pop_file : 
    pops : 
    cM : 
    r_bins : 
    bp_bins : 
    min_bp : 
    use_genotypes : 
    use_h5 : 
    stats_to_compute : 
    report : 
    report_spacing : 
    use_cache : 
    
    Recombination map has the format XXX
    pop_file has the format XXX
    """
    
    check_imports()
    
    positions, genotypes, counts, sample_ids = get_genotypes(vcf_file, bed_file=bed_file, chromosome=chromosome, min_bp=min_bp, use_h5=use_h5, report=report)
    
    if report == True:
        print("kept {0} total variants".format(len(positions))); sys.stdout.flush()
    
    if report is True: 
        print("assigning recombination rates to positions, if recombination map is passed"); sys.stdout.flush()
    
    if rec_map_file is not None and r_bins is not None:
        pos_rs = assign_recombination_rates(positions, rec_map_file, map_name=map_name, map_sep=map_sep, cM=cM, report=report)
        bins = r_bins
    else:
        if bp_bins is not None:
            bins = bp_bins
        else:
            bins = []
    
    reported_stats = get_reported_stats(genotypes, bins, sample_ids, positions=positions, pos_rs=pos_rs, pop_file=pop_file, pops=pops, use_genotypes=use_genotypes, report=report, stats_to_compute=stats_to_compute, report_spacing=report_spacing, use_cache=use_cache)
    
    return reported_stats

def bootstrap_data(all_data, normalization=['pi2_1_1_1_1','H_1_1']):
    """
    all_data : dictionary (with arbitrary keys), where each value is are ld statistics computed
               from a distinct region. all_data[reg]
               stats from each region has keys, 'bins', 'sums', 'stats', and optional 'pops' (anything else?)
    normalization : we work with sigma_d^2 statistics, and by default we use population 1 to normalize stats
    
    We first check that all 'stats', 'bins', 'pops' (if present), match across all regions
    
    If there are N total regions, we compute N bootstrap replicates by sampling N times with replacement
        and summing over all 'sums'.
    """
    
    ## Check consistencies of bins, stats, and data sizes
    
    
    
    regions = list(all_data.keys())
    reg = regions[0]
    stats = all_data[reg]['stats']
    N = len(regions)
    
    # get means
    means = [0*sums for sums in all_data[reg]['sums']]
    for reg in regions:
        for ii in range(len(means)):
            means[ii] += all_data[reg]['sums'][ii]
    
    for ii in range(len(means)-1):
        means[ii] /= means[ii][stats[0].index(normalization[0])]
    means[-1] /= means[-1][stats[1].index(normalization[1])]
    
    
    # construct bootstrap data
    bootstrap_data = [np.zeros((len(sums),N)) for sums in means] 
    
    for boot_num in range(N):
        boot_means = [0*sums for sums in means]
        samples = np.random.choice(regions, N)
        for reg in samples:
            for ii in range(len(boot_means)):
                boot_means[ii] += all_data[reg]['sums'][ii]
        
        for ii in range(len(boot_means)-1):
            boot_means[ii] /= boot_means[ii][stats[0].index(normalization[0])]
        boot_means[-1] /= boot_means[-1][stats[1].index(normalization[1])]
        
        for ii in range(len(boot_means)):
            bootstrap_data[ii][:,boot_num] = boot_means[ii]

    varcovs = [np.cov(bootstrap_data[ii]) for ii in range(len(bootstrap_data))]

    mv = {}
    mv['bins'] = all_data[reg]['bins']
    mv['stats'] = all_data[reg]['stats']
    if 'pops' in all_data[reg]:
        mv['pops'] = all_data[reg]['pops']
    mv['means'] = means
    mv['varcovs'] = varcovs
    
    return mv











