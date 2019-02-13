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

def load_h5(vcf_file):
    check_imports()
    ## open the h5 callset, create if doesn't exist
    ## saves h5 callset as same name and path, but with h5 extension instead of vcf or vcf.gz
    h5_file_path = vcf_file.split('.vcf')[0] + '.h5' # kinda hacky, sure
    try:
        callset = h5py.File(h5_file_path, mode='r')
    except OSError:
        print("creating and saving h5 file"); sys.stdout.flush()
        allel.vcf_to_hdf5(vcf_file, h5_file_path, fields='*', overwrite=True)
        callset = h5py.File(h5_file_path, mode='r')
    return callset


### genotype function
def get_genotypes(vcf_file, bed_file=None, min_bp=None, use_h5=True, report=True):
    """
    Given a vcf file, we extract the biallelic SNP genotypes.
    If bed_file is None, we use all valid variants. Otherwise we filter genotypes
        by the given bed file.
        Warning!!! make sure that the chromosome labels are consistent (e.g., chr22 vs 22)
    min_bp : only used with bed file, filters out features that are smaller than min_bp
    If use_h5 is True, we try to load the h5 file, which has the same path/name as 
        vcf_file, but with *.h5 instead of *.vcf or *.vcf.gz. If the h5 file does not
        exist, we create it and save it as *.h5
    report : prints progress updates if True, silent otherwise
    """
    
    check_imports()
    
    if use_h5 is True:
        callset = load_h5(vcf_file)
    else:
        ## read the vcf directly
        raise ValueError("Use hdf5 format.")
    
    all_genotypes = allel.GenotypeChunkedArray(callset['calldata/GT'])
    all_positions = callset['variants/POS'][:]
    
    if report is True: print("loaded genotypes"); sys.stdout.flush()
    
    # filter SNPs not in bed file, if one is given
    if bed_file is not None: # filter genotypes and positions
        mask_bed = pandas.read_csv(bed_file, sep='\t', header=None)
        chroms = ['chr'+c for c in list(set(callset['variants/CHROM'][:]))]
        
        chrom_filter = [False] * len(mask_bed)
        for chrom in chroms:
            chrom_filter = np.logical_or(chrom_filter, mask_bed[0] == chrom)
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
            map_values = rec_map[rec_map.keys()[1]]
        
    # for positions sticking out the end of the map, they take the value of the closest position
    # ideally, you'd filter these out
    
    if cM == True:
        map_values /= 100
    
    pos_rs = assign_r_pos(positions, [map_positions, map_values])
    
    return pos_rs


def g_tally_counter(g_l, g_r):
    gs = list(zip(g_l,g_r))
    c = Counter(gs)
    return (c[(2,2)], c[(2,1)], c[(2,0)], 
            c[(1,2)], c[(1,1)], c[(1,0)], 
            c[(0,2)], c[(0,1)], c[(0,0)])


def h_tally_counter(h_l, h_r):
    hs = list(zip(h_l, h_r))
    c = Counter(hs)
    return (c[(1,1)], c[(1,0)], c[(0,1)], c[(0,0)])


def count_types(genotypes, bins, sample_ids, positions=None, pos_rs=None, pop_file=None, pops=None, use_genotypes=True, report=True):
    """
    genotypes : in format of 0,1,2
    """
    
    pop_indexes = {}
    if pops is not None:
        samples = pandas.read_csv(pop_file, sep='\t')
        cols_to_keep = np.array([False]*np.shape(genotypes)[1])
        all_samples_to_keep = []
        for pop in pops:
            all_samples_to_keep += list(samples[samples['pop'] == pop]['sample'])
        
        
        for s in all_samples_to_keep:
            cols_to_keep[list(sample_ids.value).index(s)] = True
        
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
    
    type_counts = {}
    for b in bs:
        type_counts[b] = {}
    
    if pos_rs is not None:
        rs = pos_rs
    elif positions is not None:
        rs = positions
    
    ns = np.array([2 * sum(pop_indexes[pop]) for pop in pops])
    
    for ii,r in enumerate(rs[:-1]):
        if report is True:
            if ii%1000 == 0:
                print("tallied two locus counts {0} of {1} positions".format(ii, len(rs))); sys.stdout.flush()
        
        if use_genotypes == True:
            gs_ii = genotypes_pops_012[ii]
            gs_l = [gs_ii.compress(pop_indexes[pop]) for pop in pops]
        else:
            gs_ii = haplotypes_pops_01[ii]
            gs_l = [gs_ii.compress(pop_indexes_haps[pop]) for pop in pops]
        
        
        
        allele_counts = np.array([sum(g_l) for g_l in gs_l])
        
        if np.all(allele_counts == np.array([0]*len(pops))) or np.all(allele_counts == ns):
            continue
        
        for b in bs:
            filt = np.logical_and(pos_rs[ii+1:] - r >= b[0], pos_rs[ii+1:] - r < b[1])
            if use_genotypes == True:
                gs_to_right = genotypes_pops_012[ii+1:].compress(filt, axis=0)
            else:
                gs_to_right = haplotypes_pops_01[ii+1:].compress(filt, axis=0)
            
            if np.shape(gs_to_right) is ():
                continue
            
            for gs_jj in gs_to_right:
                if use_genotypes == True:
                    gs_r = [gs_jj.compress(pop_indexes[pop]) for pop in pops]
                else:
                    gs_r = [gs_jj.compress(pop_indexes_haps[pop]) for pop in pops]
                
                if use_genotypes == True:
                    cs = tuple([g_tally_counter(gl, gr) for gr,gl in zip(gs_l, gs_r)])
                else:
                    cs = tuple([h_tally_counter(gl, gr) for gr,gl in zip(gs_l, gs_r)])
                
                type_counts[b].setdefault(cs,0)
                type_counts[b][cs] += 1
    
    return type_counts


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
            alt_pop_nums = [ii,kk,jj]
            if use_genotypes == True:
                return 1./2 * sgc.Dz(Cs, pop_nums) + 1./2 * sgc.Dz(Cs, alt_pop_nums)
            else:
                return 1./2 * shc.Dz(Cs, pop_nums) + 1./2 * shc.Dz(Cs, alt_pop_nums)
    if s == 'pi2':
        ii,jj,kk,ll = pop_nums ### this below probably is incorrect for multiple pops
        if [ii,jj] == [kk,ll]:
            if use_genotypes == True:
                return sgc.pi2(Cs, pop_nums)
            else:
                return shc.pi2(Cs, pop_nums)
        else:
            alt_pop_nums = [kk,ll,ii,jj]
            if use_genotypes == True:
                return 1./2 * sgc.pi2(Cs, pop_nums) + 1./2 * sgc.pi2(Cs, alt_pop_nums)
            else:
                return 1./2 * shc.pi2(Cs, pop_nums) + 1./2 * shc.pi2(Cs, alt_pop_nums)


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
    
    if pops == None:
        raise ValueError("should pass pops...."); sys.stdout.flush()
    
    samples = pandas.read_csv(pop_file, sep='\t')

    populations = np.array(samples['pop'].value_counts().keys())

    samples.reset_index(drop=True, inplace=True)

### should use this above when counting two locus genotypes

    subpops = {
        # for each population, get the list of samples that belong to the population
        pop_iter: samples[samples['pop'] == pop_iter].index.tolist() for pop_iter in pops
    }
    
    ac_subpop = genotypes.count_alleles_subpops(subpops)
    
    Hs = {}
    for ii,pop1 in enumerate(list(subpops.keys())):
        for pop2 in list(subpops.keys())[ii:]:
            if pop1 == pop2:
                H = np.sum( 2. * ac_subpop[pop1][:,0] * ac_subpop[pop1][:,1] / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1]) / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1] - 1) )
            else:
                H = np.sum( ac_subpop[pop1][:,0] * ac_subpop[pop2][:,1] / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1]) / (ac_subpop[pop2][:,0] + ac_subpop[pop2][:,1]) + ac_subpop[pop1][:,1] * ac_subpop[pop2][:,0] / (ac_subpop[pop1][:,0] + ac_subpop[pop1][:,1]) / (ac_subpop[pop2][:,0] + ac_subpop[pop2][:,1]) )
            Hs[(pop1,pop2)] = H
    
    return Hs


def compute_ld_statistics(vcf_file, bed_file=None, rec_map_file=None, map_name=None, map_sep='\t', pop_file=None, pops=None, cM=True, r_bins=None, bp_bins=None, min_bp=None, use_genotypes=True, use_h5=True, stats_to_compute=None, report=True):

    """ testing
    vcf_file = '/Users/aragsdal/Data/Human/ThousandGenomes/genotypes/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
    bed_file = '/Users/aragsdal/Data/Human/ThousandGenomes/masks/gencode_v19_intergenic_strict_mask.flank20k.chr22.bed.gz'
    rec_map_file = '/Users/aragsdal/Data/Human/maps_b37/maps_chr.22'
    pop_file = '/Users/aragsdal/Data/Human/ThousandGenomes/genotypes/integrated_call_samples_v3.20130502.ALL.panel'
    pops = ['YRI','CEU','CHB']
    
    min_bp = 100
    use_h5 = True
    
    map_name = 'AA_Map'
    map_sep = ' '
    
    r_bins = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.002]
    
    stats = Parsing.compute_ld_statistics(vcf_file, bed_file=bed_file, rec_map_file=rec_map_file, map_name=map_name, pop_file=pop_file, pops=pops, r_bins=r_bins, min_bp=100)
    """
    
    check_imports()
    
    positions, genotypes, counts, sample_ids = get_genotypes(vcf_file, bed_file=bed_file, min_bp=min_bp, use_h5=use_h5, report=report)
    
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
    
    # now if bins is empty, we only return heterozygosity statistics
    type_counts = count_types(genotypes, bins, sample_ids, positions=positions, pos_rs=pos_rs, pop_file=pop_file, pops=pops, use_genotypes=use_genotypes, report=report)
    
    if stats_to_compute == None:
        if pops is None:
            stats_to_compute = Util.moment_names(1)
        else:
            stats_to_compute = Util.moment_names(len(pops))
    
    statistics_cache = cache_ld_statistics(type_counts, stats_to_compute[0], bins, use_genotypes=use_genotypes, report=report)
    
    bs = list(zip(bins[:-1],bins[1:]))
    sums = {}
    for b in bs:
        sums[b] = {}
        for stat in stats_to_compute[0]:
            sums[b][stat] = 0
            for cs in type_counts[b]:
                sums[b][stat] += type_counts[b][cs] * statistics_cache[cs][stat]
    
    Hs = get_H_statistics(genotypes, sample_ids, pop_file=pop_file, pops=pops)
    
    reported_stats = {}
    reported_stats['bins'] = bs
    reported_stats['sums'] = [np.empty(len(stats_to_compute[0])) for b in bs] + [np.empty(len(stats_to_compute[1]))]
    for ii,b in enumerate(bs):
        for s in stats_to_compute[0]:
            reported_stats['sums'][ii][stats_to_compute[0].index(s)] = sums[b][s]
    for s in stats_to_compute[1]:
        reported_stats['sums'][-1][stats_to_compute[1].index(s)] = Hs[(pops[int(s.split('_')[1])-1],pops[int(s.split('_')[2])-1])]
    reported_stats['stats'] = stats_to_compute
    
    return reported_stats

def bootstrap_data():
    pass
