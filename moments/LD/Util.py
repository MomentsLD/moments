import numpy as np


"""
Heterozygosity and LD stat names
"""

### handling data (splits, marginalization, admixture, ...) 
def het_names(num_pops):
    Hs = []
    for ii in range(num_pops):
        for jj in range(ii,num_pops):
            Hs.append('H_{0}_{1}'.format(ii+1,jj+1))
    return Hs

def ld_names(num_pops):
    Ys = []
    for ii in range(num_pops):
        for jj in range(ii,num_pops):
            Ys.append('DD_{0}_{1}'.format(ii+1,jj+1))
    for ii in range(num_pops):
        for jj in range(num_pops):
            for kk in range(jj,num_pops):
                Ys.append('Dz_{0}_{1}_{2}'.format(ii+1,jj+1,kk+1))
    for ii in range(num_pops):
        for jj in range(ii,num_pops):
            for kk in range(ii,num_pops):
                for ll in range(kk,num_pops):
                    if kk == ii == ll and jj != ii:
                        continue
                    if ii == kk and ll < jj:
                        continue
                    Ys.append('pi2_{0}_{1}_{2}_{3}'.format(ii+1,jj+1,kk+1,ll+1))
    return Ys

def moment_names(num_pops):
    """
    num_pops : number of populations, indexed [1,...,num_pops]
    """
    hn = het_names(num_pops)
    yn = ld_names(num_pops)
    return (yn, hn)

"""
We need to map moments for split and rearrangement functions
"""
mom_map = {}
def map_moment(mom):
    """
    
    """
    try:
        return mom_map[mom]
    except KeyError:
        if mom.split('_')[0] == 'DD':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'DD_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'Dz':
            popD = mom.split('_')[1]
            popsz = sorted([int(p) for p in mom.split('_')[2:]])
            mom_out = 'Dz_'+popD+'_'+'_'.join([str(p) for p in popsz])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'pi2':
            popsp = sorted([int(p) for p in mom.split('_')[1:3]])
            popsq = sorted([int(p) for p in mom.split('_')[3:]])
            ## pi2_2_2_1_1 -> pi2_1_1_2_2, pi2_1_2_1_1 -> pi2_1_1_1_2, pi2_2_2_1_3 -> pi2_1_3_2_2
            if popsp[0] > popsq[0]: # switch them
                mom_out = 'pi2_'+'_'.join([str(p) for p in popsq])+'_'+'_'.join([str(p) for p in popsp])
            elif popsp[0] == popsq[0] and popsp[1] > popsq[1]: # switch them
                mom_out = 'pi2_'+'_'.join([str(p) for p in popsq])+'_'+'_'.join([str(p) for p in popsp])
            else:
                mom_out = 'pi2_'+'_'.join([str(p) for p in popsp])+'_'+'_'.join([str(p) for p in popsq])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'H':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'H_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        else:
            mom_out = mom
        mom_map[mom] = mom_out
        return mom_map[mom]

