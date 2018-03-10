import numpy as np
from moments.LD import Numerics
from moments.LD.LDstats_mod import LDstats

"""
contains correction factors for finite sample for either single or multiple population statistics
corrections are implemented for single pop models up to order D^6 (evens)
and for multipopulation models for 2nd order statistics
"""

def corrected_onepop(stats, n=None, order=2):
    if n == None:
        return stats
    if order == 2:
        return LDstats(order2correction(n, stats.data),order=order)
    if order == 4:
        return LDstats(order4correction(n, stats.data),order=order)
    if order == 6:
        return LDstats(order6correction(n, stats.data),order=order)
    else:
        print("Haven't implemented corrections for order {0}".format(order))
        return stats

def corrected_onepop_genotypes(stats, n=None, order=2):
    """
    correct the expectations for genotype data
    if n is None, there is no sampling correction
    otherwise n 
    NOTE: unlike the correction for haploid sampling, where n is the number of chromosomes,
          here n is the diploid sample size, so we'd equilivalently have 2n haploid samples
    """
    if order == 2:
        return LDstats(order2correction_genotypes(stats.data, n), order=order)
    elif order == 4:
        return LDstats(order4correction_genotypes(stats.data, n), order=order)
    else:
        print("Haven't implemented corrections for order {0}".format(order))
        return stats

def corrected_multipop(stats, ns=None, num_pops=2):
    """
    for num_pops number of populations (currently implemented up to 4 populations as of 10/17)
    ns = (n1,n2,...) in the correct order for the list stats
    stats start of with [D1**2, D1 D2, ..., D2**2, D2 D3, ..., etc]
    """
    if ns == None:
        return stats
    
    if len(ns) != num_pops:
        raise ValueError("number of sample sizes must equal number of populations")
    
    stat_names = Numerics.moment_list(num_pops)
    if len(stat_names) != len(stats)-1:
        raise ValueError("mismatch of input moments and number of populations")
    
    corrected = np.ones(len(stats))
    for ii,name in zip(range(len(stat_names)),stat_names):
        corrected[ii] = adjust_moment(name, stat_names, stats, ns)
    
    return LDstats(corrected, num_pops=num_pops, order=2)
        
def corrected_multipop_genotypes(stats, ns=None, num_pops=2):
    """
    """
    if ns == None:
        return stats
    
    if len(ns) != num_pops:
        raise ValueError("length of ns must equal number of populations")
    
    stat_names = Numerics.moment_list(num_pops)
    if len(stat_names) != len(stats)-1:
        raise ValueError("mismatch of input moments and number of populations")

    corrected = np.ones(len(stats))
    for ii,name in zip(range(len(stat_names)),stat_names):
        corrected[ii] = adjust_moment_genotype(name, stat_names, stats, ns)
    
    return LDstats(corrected, num_pops=num_pops, order=2)

### below are all the sampling bias corrections for each statistic

def adjust_moment(name, stat_names, stats, sample_sizes):
    moment_type = name.split('_')[0]
    if moment_type == 'DD':
        popA = name.split('_')[1]
        popB = name.split('_')[2]
        if popA == popB:
            n1 = sample_sizes[int(popA)-1] # note that pop names are 1-indexed, not 0-indexed
            # need D_A^2, Dz_A, pi_A, which relies on (1-2p1)^2(1-2q1)^2, (1-2p1)^2, (1-2q1)^2, and 1
            mom1 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{0}'.format(popA))[0]]
            mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{0}'.format(popA))[0]]
            mom3 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{0}_{0}'.format(popA))[0]]
            mom4 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{0}'.format(popA))[0]]
            mom5 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{0}'.format(popA))[0]]
            return mom1 * (-2 + 4*n1 - 3*n1**2 + n1**3)/n1**3 + mom2 * (-1 + n1)**2/n1**3 + mom3 * (-1 + n1)/(16.*n1**2) - mom4 * (-1 + n1)/(16.*n1**2) - mom5 * (-1 + n1)/(16.*n1**2) + 1. * (-1 + n1)/(16.*n1**2)
        else:
            n1 = sample_sizes[int(popA)-1]
            n2 = sample_sizes[int(popB)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popA,popB))[0]]
            return mom1 * ((-1 + n1)*(-1 + n2))/(n1*n2)
    elif moment_type == 'Dz':
        popD = name.split('_')[1]
        popp = name.split('_')[2]
        popq = name.split('_')[3]
        if popD == popp == popq:
            n1 = sample_sizes[int(popD)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{0}'.format(popD))[0]]
            return mom1 * ((-2 + n1)**2*(-1 + n1))/n1**3 + mom2 * (4*(2 - 3*n1 + n1**2))/n1**3
        elif popD == popp or popD == popq:
            n1 = sample_sizes[int(popD)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            return mom1 * (2 - 3*n1 + n1**2)/n1**2
        elif popp == popq:
            n1 = sample_sizes[int(popD)-1]
            n2 = sample_sizes[int(popp)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            try:
                mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popD,popp))[0]]
            except IndexError:
                mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popp,popD))[0]]
            return mom1 * (-1 + n1)/n1 + mom2 * (4*(-1 + n1))/(n1*n2)
        else:
            n1 = sample_sizes[int(popD)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            return mom1 * (-1 + n1)/n1
    elif moment_type == 'zz':
        popp1 = name.split('_')[1]
        popp2 = name.split('_')[2]
        popq1 = name.split('_')[3]
        popq2 = name.split('_')[4]
        if popp1 == popp2 == popq1 == popq2: # e.g. 1_1_1_1
            n1 = sample_sizes[int(popp1)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{0}'.format(popp1))[0]]
            mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{0}'.format(popp1))[0]]
            mom3 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{0}_{0}'.format(popp1))[0]]
            mom4 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{0}'.format(popp1))[0]]
            mom5 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{0}'.format(popp1))[0]]
            return mom1 * (32*(-1 + n1))/n1**3 + mom2 * (16*(-1 + n1)**2)/n1**3 + mom3 * (-1 + n1)**2/n1**2 + mom4 * (-1 + n1)/n1**2 + mom5 * (-1 + n1)/n1**2 + 1. * n1**(-2)
        elif popp1 == popp2: 
            if popq1 == popp1: # e.g. 1_1_1_2
                n1 = sample_sizes[int(popp1)-1]
                n2 = sample_sizes[int(popq2)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{1}'.format(popp1,popq2))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{0}_{1}'.format(popp1,popq2))[0]]
                mom3 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * (8*(-1 + n1))/n1**2 + mom2 * (-1 + n1)/n1 + mom3 * 1/n1
            elif popq2 == popp1: # e.g. 2_2_1_2
                n1 = sample_sizes[int(popp1)-1]
                n2 = sample_sizes[int(popq1)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{1}'.format(popp1,popq1))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{1}_{0}'.format(popp1,popq1))[0]]
                mom3 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * (8*(-1 + n1))/n1**2 + mom2 * (-1 + n1)/n1 + mom3 * 1/n1
            elif popq1 == popq2: # e.g. 1_1_2_2
                n1 = sample_sizes[int(popp1)-1]
                n2 = sample_sizes[int(popq1)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{1}_{1}'.format(popp1,popq1))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                mom3 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * ((-1 + n1)*(-1 + n2))/(n1*n2) + mom2 * (-1 + n1)/(n1*n2) + mom3 * (-1 + n2)/(n1*n2) + 1. * 1/(n1*n2)
            else: # e.g. 1_1_2_3
                n1 = sample_sizes[int(popp1)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{1}_{2}'.format(popp1,popq1,popq2))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * (-1 + n1)/n1 + mom2 * 1/n1
        else: # popp1 != popp2
            if popq1 == popq2:
                if popp1 == popq1: # e.g. 1_2_1_1
                    n1 = sample_sizes[int(popp1)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{0}_{0}'.format(popp1,popp2))[0]]
                    mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{0}'.format(popp1,popp2))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                    return mom1 * (-1 + n1)/n1 + mom2 * (8*(-1 + n1))/n1**2 + mom3 * 1/n1
                elif popp2 == popq1: # e.g. 1_2_2_2
                    n1 = sample_sizes[int(popp2)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{1}_{1}'.format(popp1,popp2))[0]]
                    mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{0}'.format(popp2,popp1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                    return mom1 * (-1 + n1)/n1 + mom2 * (8*(-1 + n1))/n1**2 + mom3 * 1/n1
                else: # e.g. 1_2_3_3
                    n3 = sample_sizes[int(popq1)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{2}'.format(popp1,popp2,popq1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                    return mom1 * (-1 + n3)/n3 + mom2 * 1/n3
            else: # popq1 != popq2
                if popp1 == popq1:
                    if popp2 == popq2: # e.g. 1_2_1_2
                        n1 = sample_sizes[int(popp1)-1]
                        n2 = sample_sizes[int(popp2)-1]
                        mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{0}_{1}'.format(popp1,popp2))[0]]
                        mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popp1,popp2))[0]]
                        mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{1}'.format(popp1,popp2))[0]]
                        mom4 = stats[np.argwhere(np.array(stat_names) == 'Dz_{1}_{0}_{0}'.format(popp1,popp2))[0]]
                        return mom1 + mom2 * 16/(n1*n2) + mom3 * 4/n1 + mom4 * 4/n2
                    else: # e.g. 1_3_1_2 or 1_2_1_3
                        n1 = sample_sizes[int(popp1)-1]
                        mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{0}_{2}'.format(popp1,popp2,popq2))[0]]
                        mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp1,popp2,popq2))[0]]
                        return mom1 + mom2 * 4/n1
                elif popp1 == popq2: # e.g. 2_3_1_2
                    n2 = sample_sizes[int(popp1)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{0}'.format(popp1,popp2,popq1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp1,popp2,popq1))[0]]
                    return mom1 + mom2 * 4/n2
                elif popp2 == popq2: # e.g. 1_3_2_3
                    n3 = sample_sizes[int(popp2)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{1}'.format(popp1,popp2,popq1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp2,popp1,popq1))[0]]
                    return mom1 + mom2 * 4/n3
                elif popp2 == popq1: # e.g. 1_2_2_3
                    n2 = sample_sizes[int(popp2)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{1}_{2}'.format(popp1,popp2,popq2))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp2,popp1,popq2))[0]]
                    return mom1 + mom2 * 4/n2
                else: # e.g. 1_2_3_4
                    mom = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{3}'.format(popp1,popp2,popq1,popq2))[0]]
                    return mom
    elif moment_type == 'zp':
        popp1 = name.split('_')[1]
        popp2 = name.split('_')[2]
        if popp1 == popp2:
            n1 = sample_sizes[int(popp1)-1]
            mom = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{0}'.format(popp1))[0]]
            return mom * (-1 + n1)/n1 + 1./n1
        else:
            mom = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
            return mom
    elif moment_type == 'zq':
        popq1 = name.split('_')[1]
        popq2 = name.split('_')[2]
        if popq1 == popq2:
            n1 = sample_sizes[int(popq1)-1]
            mom = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{0}'.format(popq1))[0]]
            return mom * (-1 + n1)/n1 + 1./n1
        else:
            mom = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
            return mom
    else:
        return -1e6


def order2correction(n, stats):
    stat_names = Numerics.moment_names_onepop(2)
    return np.array([ adjust_D2(n, stat_names, stats),
                      adjust_Dz(n, stat_names, stats),
                      adjust_pi(n, stat_names, stats),
                      adjust_s1(n, stat_names, stats),
                      1])

def order4correction(n, stats):
    stat_names = Numerics.moment_names_onepop(4)
    return np.array([ adjust_D4(n, stat_names, stats),
                      adjust_D3z(n, stat_names, stats),
                      adjust_D2pi(n, stat_names, stats),
                      adjust_Dpiz(n, stat_names, stats),
                      adjust_pi2(n, stat_names, stats),
                      adjust_D2s1(n, stat_names, stats),
                      adjust_Dzs1(n, stat_names, stats),
                      adjust_pis1(n, stat_names, stats),
                      adjust_s2(n, stat_names, stats),
                      adjust_D2(n, stat_names, stats),
                      adjust_Dz(n, stat_names, stats),
                      adjust_pi(n, stat_names, stats),
                      adjust_s1(n, stat_names, stats),
                      1])

def order6correction(n, stats):
    stat_names = Numerics.moment_names_onepop(6)
    return np.array([ adjust_D6(n, stat_names, stats),
                      adjust_D5z(n, stat_names, stats),
                      adjust_D4pi(n, stat_names, stats),
                      adjust_D3piz(n, stat_names, stats),
                      adjust_D2pi2(n, stat_names, stats),
                      adjust_Dpi2z(n, stat_names, stats),
                      adjust_pi3(n, stat_names, stats),
                      adjust_D4s1(n, stat_names, stats),
                      adjust_D3zs1(n, stat_names, stats),
                      adjust_D2pis1(n, stat_names, stats),
                      adjust_Dpizs1(n, stat_names, stats),
                      adjust_pi2s1(n, stat_names, stats),
                      adjust_D2s2(n, stat_names, stats),
                      adjust_Dzs2(n, stat_names, stats),
                      adjust_pis2(n, stat_names, stats),
                      adjust_s3(n, stat_names, stats),
                      
                      adjust_D4(n, stat_names, stats),
                      adjust_D3z(n, stat_names, stats),
                      adjust_D2pi(n, stat_names, stats),
                      adjust_Dpiz(n, stat_names, stats),
                      adjust_pi2(n, stat_names, stats),
                      adjust_D2s1(n, stat_names, stats),
                      adjust_Dzs1(n, stat_names, stats),
                      adjust_pis1(n, stat_names, stats),
                      adjust_s2(n, stat_names, stats),
                      adjust_D2(n, stat_names, stats),
                      adjust_Dz(n, stat_names, stats),
                      adjust_pi(n, stat_names, stats),
                      adjust_s1(n, stat_names, stats),
                      1])

def adjust_s1(n, stat_names, stats):
    s1 = stats[stat_names.index('1_s1')]
    return (n-1.)/n * s1

def adjust_D2(n, stat_names, stats):
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return D2 * (-2 + 4*n - 3*n**2. + n**3.)/n**3. + Dz * (1 - 2*n + n**2.)/n**3. + pi * (-n + n**2.)/n**3.

def adjust_Dz(n, stat_names, stats):
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    return D2 * (8 - 12*n + 4*n**2.)/n**3. + Dz * (-4 + 8*n - 5*n**2. + n**3.)/n**3.

def adjust_pi(n, stat_names, stats):
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return D2 * (-2 + 2*n)/n**3. + Dz * (1 - 2*n + n**2.)/n**3. + pi * (n - 2*n**2. + n**3.)/n**3.

def adjust_s2(n, stat_names, stats):
    s2 = stats[stat_names.index('1_s2')]
    s1 = stats[stat_names.index('1_s1')]
    return s2 * (-6 + 11*n - 6*n**2. + n**3.)/n**3. + s1 * (-1 + n)**2/n**3.

def adjust_D2s1(n, stat_names, stats):
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs2 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D2s1 * (-72 + 168*n - 144*n**2. + 59*n**3. - 12*n**4. + n**5.)/n**5. +
            Dzs2 * ((-2 + n)**2*(3 - 4*n + n**2.))/n**5. +
            pis1 * (-6 + 11*n - 6*n**2. + n**3.)/n**4. +
            D2 * (4*(7 - 17*n + 15*n**2. - 6*n**3. + n**4.))/n**5. +
            Dz * (2*(-1 + n)**3)/n**5. +
            pi * (2*(-1 + n)**2)/n**4.][0]

def adjust_Dzs1(n, stat_names, stats):
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs2 = stats[stat_names.index('D^1_z_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    return [D2s1 * (12*(24 - 50*n + 35*n**2. - 10*n**3. + n**4.))/n**5. +
            Dzs2 * ((-2 + n)**2*(-12 + 19*n - 8*n**2. + n**3.))/n**5. +
            D2 * (-112 + 240*n - 172*n**2. + 48*n**3. - 4*n**4.)/n**5. +
            Dz * (8 - 24*n + 26*n**2. - 12*n**3. + 2*n**4.)/n**5.][0]

def adjust_pis1(n, stat_names, stats):
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D2s1 * (12*(-6 + 11*n - 6*n**2. + n**3.))/n**5. + 
            Dzs1 * (2*(-1 + n)**2*(6 - 5*n + n**2.))/n**5. + 
            pis1 * ((-1 + n)**2*(6 - 5*n + n**2.))/n**4. + 
            D2 * (-4*(-7 + 13*n - 7*n**2. + n**3.))/n**5. + 
            Dz * (2*(-1 + n)**3)/n**5. +
            pi * (2*(-1 + n)**3)/n**4.][0]

def adjust_D4(n, stat_names, stats):
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D4 * (-144 + 408*n - 480*n**2. + 324*n**3. - 146*n**4. + 47*n**5. - 10*n**6. + n**7.)/n**7. + 
            D3z * (6*(-2 + n)**2*(18 - 33*n + 21*n**2. - 7*n**3. + n**4.))/n**7. + 
            D2pi * (6*(-432 + 1152*n - 1176*n**2. + 580*n**3. - 133*n**4. + 8*n**5. + n**6.))/n**7. + 
            Dpiz * (2*(72 - 120*n + 20*n**2. + 55*n**3. - 32*n**4. + 5*n**5.))/n**7. + 
            pi2 * (3*(-36 + 60*n - 25*n**2. + n**4.))/n**6. + 
            D2s1 * (-12*(-42 + 119*n - 131*n**2. + 71*n**3. - 19*n**4. + 2*n**5.))/n**7. + 
            Dzs1 * (-2*(-1 + n)**2*(6 - 5*n + n**2.))/n**7. + 
            pis1 * (-3*(-6 + 11*n - 6*n**2. + n**3.))/n**6. + 
            D2 * (-98 + 294*n - 346*n**2. + 202*n**3. - 59*n**4. + 7*n**5.)/n**7. + 
            Dz * (-1 + n)**4/n**7. + 
            pi * (-3 + 6*n - 4*n**2. + n**3.)/n**6.][0]

def adjust_D3z(n, stat_names, stats):
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D4 * 1.*(4.*(144 - 372*n + 378*n**2. - 204*n**3. + 65*n**4. - 12*n**5. + n**6.))/n**7. + 
            D3z * 1.*(-1728 + 4608*n - 4836*n**2. + 2658*n**3. - 846*n**4. + 161*n**5. - 18*n**6. + n**7.)/n**7. + 
            D2pi * 1.*(12*(864 - 2304*n + 2394*n**2. - 1273*n**3. + 374*n**4. - 59*n**5. + 4*n**6.))/n**7. + 
            Dpiz * 1.*(-576 + 1248*n - 892*n**2. + 204*n**3. + 37*n**4. - 24*n**5. + 3*n**6.)/n**7. + 
            pi2 * 1.*(16*(-3 + n)**2*(2 - 3*n + n**2.))/n**6. + 
            D2s1 * 1.*(-6*(-2 + n)**2*(84 - 151*n + 86*n**2. - 21*n**3. + 2*n**4.))/n**7. + 
            Dzs1 * -1.*(((-2 + n)**2*(-12. + 25*n - 16*n**2. + 3*n**3.))/n**7.) + 
            pis1 * 1.*(-4*(-2 + n)**2*(3 - 4*n + n**2.))/n**6. + 
            D2 * 1.*((-2 + n)**3*(-49 + 70*n - 24*n**2. + 3*n**3.))/n**7. + 
            Dz * 1.*((-2 + n)**2*(-1 + n)**3)/n**7. + 
            pi * 1.*((-2 + n)**3*(-1 + n))/n**6.][0]

def adjust_D2pi(n, stat_names, stats):
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D4 * (2*(-72 + 168*n - 144*n**2. + 59*n**3. - 12*n**4. + n**5.))/n**7. + 
            D3z * ((-3 + n)**2*(48 - 88*n + 50*n**2. - 11*n**3. + n**4.))/n**7. + 
            D2pi * (-2592 + 6912*n - 7212*n**2. + 3892*n**3. - 1191*n**4. + 211*n**5. - 21*n**6. + n**7.)/n**7. + 
            Dpiz * ((-3 + n)**2*(16 - 32*n + 22*n**2. - 7*n**3. + n**4.))/n**7. + 
            pi2 * ((-1 + n)*(6 - 5*n + n**2.)**2)/n**6. + 
            D2s1 * (2*(-2 + n)**2*(63 - 111*n + 60*n**2. - 13*n**3. + n**4.))/n**7. + 
            Dzs1 * ((-3 + n)*(2 - 3*n + n**2.)**2)/n**7. + 
            pis1 * ((-1 + n)**2*(6 - 5*n + n**2.))/n**6. + 
            D2 * (-98 + 280*n - 309*n**2. + 167*n**3. - 45*n**4. + 5*n**5.)/n**7. + 
            Dz * (-1 + n)**4/n**7. + 
            pi * (-1 + n)**3/n**6.][0]

def adjust_Dpiz(n, stat_names, stats):
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D4 * (24*(24 - 50*n + 35*n**2. - 10*n**3. + n**4.))/n**7. + 
            D3z * (18*(-4 + n)**2*(-6 + 11*n - 6*n**2. + n**3.))/n**7. + 
            D2pi * (36*(12 - 7*n + n**2.)**2*(2 - 3*n + n**2.))/n**7. + 
            Dpiz * ((-1 + n)*(-24 + 26*n - 9*n**2. + n**3.)**2)/n**7. + 
            D2s1 * (-6*(-2 + n)**2*(84 - 145*n + 75*n**2. - 15*n**3. + n**4.))/n**7. + 
            Dzs1 * ((12 - 7*n + n**2.)*(2 - 3*n + n**2.)**2)/n**7. + 
            D2 * ((-7 + n)**2*(-2 + n)**3*(-1 + n))/n**7. + 
            Dz * ((-2 + n)**2*(-1 + n)**3)/n**7.][0]

def adjust_pi2(n, stat_names, stats):
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D4 * (24*(-6 + 11*n - 6*n**2. + n**3.))/n**7. + 
            D3z * (24*(-3 + n)**2*(2 - 3*n + n**2.))/n**7. + 
            D2pi * (72*(-1 + n)*(6 - 5*n + n**2.)**2)/n**7. + 
            Dpiz * (4*(-6 + 11*n - 6*n**2. + n**3.)**2)/n**7. + 
            pi2 * (-6 + 11*n - 6*n**2. + n**3.)**2/n**6. + 
            D2s1 * (-12*(-42 + 113*n - 114*n**2. + 54*n**3. - 12*n**4. + n**5.))/n**7. + 
            Dzs1 * (2*(-1 + n)**3*(6 - 5*n + n**2.))/n**7. + 
            pis1 * ((-1 + n)**3*(6 - 5*n + n**2.))/n**6. + 
            D2 * (2*(-1 + n)*(7 - 6*n + n**2.)**2)/n**7. + 
            Dz * (-1 + n)**4/n**7. + 
            pi * (-1 + n)**4/n**6.][0]

## order 6 adjustments

def adjust_D6(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (-86400 + 283680*n - 402480*n**2 + 336240*n**3 - 189480*n**4 + 78060*n**5 - 24834*n**6 + 6334*n**7 - 1305*n**8 + 205*n**9 - 21*n**10 + n**11)/n**11 +
            D5z * (15*(43200 - 141840*n + 199800*n**2 - 163392*n**3 + 88032*n**4 - 33426*n**5 + 9259*n**6 - 1878*n**7 + 268*n**8 - 24*n**9 + n**10))/n**11 +
            D4pi * (15*(-864000 + 2793600*n - 3828240*n**2 + 2985816*n**3 - 1487400*n**4 + 497760*n**5 - 112826*n**6 + 16659*n**7 - 1415*n**8 + 45*n**9 + n**10))/n**11 +
            D3piz * (10*(432000 - 1324800*n + 1676280*n**2 - 1160700*n**3 + 482676*n**4 - 121336*n**5 + 16635*n**6 - 655*n**7 - 111*n**8 + 11*n**9))/n**11 +
            D2pi2 * (15*(-432000 + 1108800*n - 1035480*n**2 + 391560*n**3 + 7318*n**4 - 57228*n**5 + 19675*n**6 - 2775*n**7 + 127*n**8 + 3*n**9))/n**11 +
            Dpi2z * (3*(43200 - 24480*n - 113892*n**2 + 162612*n**3 - 85465*n**4 + 19140*n**5 - 838*n**6 - 312*n**7 + 35*n**8))/n**11 +
            pi3 * (5*(-14400 + 22560*n - 3796*n**2 - 8328*n**3 + 4835*n**4 - 915*n**5 + 41*n**6 + 3*n**7))/n**10 +
            D4s1 * (-60*(-46800 + 153660*n - 214410*n**2 + 170550*n**3 - 86743*n**4 + 29686*n**5 - 6910*n**6 + 1060*n**7 - 97*n**8 + 4*n**9))/n**11 +
            D3zs1 * (-30*(21600 - 70920*n + 97620*n**2 - 74690*n**3 + 35091*n**4 - 10430*n**5 + 1920*n**6 - 200*n**7 + 9*n**8))/n**11 +
            D2pis1 * (-15*(-129600 + 360720*n - 387360*n**2 + 199920*n**3 - 43568*n**4 - 3045*n**5 + 3565*n**6 - 675*n**7 + 43*n**8))/n**11 +
            Dpizs1 * (-3*(7200 - 8760*n - 9916*n**2 + 22726*n**3 - 15565*n**4 + 5065*n**5 - 799*n**6 + 49*n**7))/n**11 +
            pi2s1 * (-5*(-3600 + 6180*n - 2092*n**2 - 1275*n**3 + 995*n**4 - 225*n**5 + 17*n**6))/n**10 +
            D2s2 * (30*(-3720 + 12214*n - 16429*n**2 + 11802*n**3 - 4900*n**4 + 1176*n**5 - 151*n**6 + 8*n**7))/n**11 +
            Dzs2 * (3*(-1 + n)**2*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**11 +
            pis2 * (5*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5))/n**10 +
            s3 * 0 +
            D4 * (5*(-121680 + 405600*n - 576000*n**2 + 466914*n**3 - 242156*n**4 + 84558*n**5 - 20116*n**6 + 3167*n**7 - 300*n**8 + 13*n**9))/n**11 +
            D3z * (15*(6480 - 22680*n + 33630*n**2 - 27929*n**3 + 14337*n**4 - 4693*n**5 + 963*n**6 - 114*n**7 + 6*n**8))/n**11 +
            D2pi * (15*(-38880 + 116640*n - 140340*n**2 + 87484*n**3 - 29369*n**4 + 4467*n**5 + 113*n**6 - 127*n**7 + 12*n**8))/n**11 +
            Dpiz * (3600 - 6720*n + 346*n**2 + 7599*n**3 - 7365*n**4 + 3145*n**5 - 661*n**6 + 56*n**7)/n**11 +
            pi2 * (5*(-900 + 1680*n - 829*n**2 - 81*n**3 + 176*n**4 - 51*n**5 + 5*n**6))/n**10 +
            D2s1 * (-30*(-1116 + 3906*n - 5696*n**2 + 4521*n**3 - 2121*n**4 + 591*n**5 - 91*n**6 + 6*n**7))/n**11 +
            Dzs1 * -(((-1 + n)**2*(60 - 116*n + 89*n**2 - 31*n**3 + 4*n**4))/n**11.) +
            pis1 * (-5*(-30 + 73*n - 69*n**2 + 34*n**3 - 9*n**4 + n**5))/n**10 +
            s2 * 0 +
            D2 * (-1922 + 7688*n - 12981*n**2 + 12035*n**3 - 6635*n**4 + 2181*n**5 - 397*n**6 + 31*n**7)/n**11 +
            Dz * (-1 + n)**6/n**11 +
            pi * (-5 + 15*n - 20*n**2 + 15*n**3 - 6*n**4 + n**5)/n**10 +
            s1 * 0][0]
    
def adjust_D5z(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (4*(86400 - 269280*n + 355200*n**2 - 269160*n**3 + 133440*n**4 - 46480*n**5 + 11824*n**6 - 2215*n**7 + 295*n**8 - 25*n**9 + n**10))/n**11 +
            D5z * (-2592000 + 8164800*n - 10867680*n**2 + 8255280*n**3 - 4051320*n**4 + 1370760*n**5 - 330920*n**6 + 57694*n**7 - 7205*n**8 + 625*n**9 - 35*n**10 + n**11)/n**11 +
            D4pi * (40*(1296000 - 4082400*n + 5407920*n**2 - 4049736*n**3 + 1928916*n**4 - 618870*n**5 + 137131*n**6 - 20979*n**7 + 2149*n**8 - 135*n**9 + 4*n**10))/n**11 +
            D3piz * (10*(-1728000 + 5299200*n - 6743520*n**2 + 4761680*n**3 - 2081864*n**4 + 589008*n**5 - 108016*n**6 + 12225*n**7 - 721*n**8 + 7*n**9 + n**10))/n**11 +
            D2pi2 * (20*(1296000 - 3542400*n + 3776040*n**2 - 2016900*n**3 + 535514*n**4 - 33607*n**5 - 20005*n**6 + 6035*n**7 - 709*n**8 + 32*n**9))/n**11 +
            Dpi2z * (-518400 + 639360*n + 580464*n**2 - 1390320*n**3 + 967960*n**4 - 335160*n**5 + 61111*n**6 - 5055*n**7 + 25*n**8 + 15*n**9)/n**11 +
            pi3 * (32*(-5 + n)**2*(288 - 480*n + 170*n**2 + 55*n**3 - 38*n**4 + 5*n**5))/n**10 +
            D4s1 * (-20*(561600 - 1793520*n + 2414640*n**2 - 1839780*n**3 + 891660*n**4 - 290908*n**5 + 65483*n**6 - 10165*n**7 + 1055*n**8 - 67*n**9 + 2*n**10))/n**11 +
            D3zs1 * (-10*(-259200 + 843840*n - 1153800*n**2 + 882680*n**3 - 420850*n**4 + 130826*n**5 - 26725*n**6 + 3485*n**7 - 265*n**8 + 9*n**9))/n**11 +
            D2pis1 * (-20*(388800 - 1136160*n + 1337580*n**2 - 833160*n**3 + 295461*n**4 - 56698*n**5 + 3660*n**6 + 650*n**7 - 141*n**8 + 8*n**9))/n**11 +
            Dpizs1 * (86400 - 155520*n + 11448*n**2 + 153340*n**3 - 145570*n**4 + 62725*n**5 - 14473*n**6 + 1735*n**7 - 85*n**8)/n**11 +
            pi2s1 * (-8*(7200 - 15960*n + 11804*n**2 - 2830*n**3 - 565*n**4 + 425*n**5 - 79*n**6 + 5*n**7))/n**10 +
            D2s2 * (10*(-2 + n)**2*(11160 - 26322*n + 23383*n**2 - 10443*n**3 + 2521*n**4 - 315*n**5 + 16*n**6))/n**11 +
            Dzs2 * ((-2 + n)**2*(-360 + 942*n - 889*n**2 + 373*n**3 - 71*n**4 + 5*n**5))/n**11 +
            pis2 * (8*(-2 + n)**2*(60 - 107*n + 59*n**2 - 13*n**3 + n**4))/n**10 +
            s3 * 0 +
            D4 * (10*(243360 - 787800*n + 1077540*n**2 - 834904*n**3 + 411452*n**4 - 136378*n**5 + 31145*n**6 - 4897*n**7 + 514*n**8 - 33*n**9 + n**10))/n**11 +
            D3z * (5*(-77760 + 267840*n - 390780*n**2 + 320454*n**3 - 164032*n**4 + 54708*n**5 - 11968*n**6 + 1669*n**7 - 136*n**8 + 5*n**9))/n**11 +
            D2pi * (20*(116640 - 362880*n + 465450*n**2 - 325451*n**3 + 136087*n**4 - 34485*n**5 + 4933*n**6 - 282*n**7 - 14*n**8 + 2*n**9))/n**11 +
            Dpiz * (-14400 + 34080*n - 22564*n**2 - 6520*n**3 + 16775*n**4 - 9805*n**5 + 2824*n**6 - 415*n**7 + 25*n**8)/n**11 +
            pi2 * (2*(7200 - 17040*n + 14432*n**2 - 5200*n**3 + 450*n**4 + 215*n**5 - 62*n**6 + 5*n**7))/n**10 +
            D2s1 * (-10*(-2 + n)**2*(3348 - 8529*n + 8408*n**2 - 4235*n**3 + 1166*n**4 - 168*n**5 + 10*n**6))/n**11 +
            Dzs1 * -(((-2 + n)**2*(-60 + 191*n - 237*n**2 + 144*n**3 - 43*n**4 + 5*n**5))/n**11.) +
            pis1 * (-2*(-2 + n)**2*(60 - 116*n + 77*n**2 - 24*n**3 + 3*n**4))/n**10 +
            s2 * 0 +
            D2 * ((-2 + n)**3*(-961 + 2387*n - 2190*n**2 + 939*n**3 - 190*n**4 + 15*n**5))/n**11 +
            Dz * ((-2 + n)**2*(-1 + n)**5)/n**11 +
            pi * ((-2 + n)**3*(-2 + 4*n - 3*n**2 + n**3))/n**10 +
            s1 * 0][0]
    

def adjust_D4pi(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (2*(-43200 + 127440*n - 155400*n**2 + 105768*n**3 - 45408*n**4 + 13054*n**5 - 2565*n**6 + 337*n**7 - 27*n**8 + n**9))/n**11 +
            D5z * (648000 - 1954800*n + 2449800*n**2 - 1717872*n**3 + 760104*n**4 - 225354*n**5 + 46039*n**6 - 6498*n**7 + 616*n**8 - 36*n**9 + n**10)/n**11 +
            D4pi * (-12960000 + 39744000*n - 50795280*n**2 + 36365736*n**3 - 16413792*n**4 + 4954128*n**5 - 1029806*n**6 + 148963*n**7 - 14918*n**8 + 1012*n**9 - 44*n**10 + n**11)/n**11 +
            D3piz * (2*(2160000 - 6624000*n + 8455800*n**2 - 6031700*n**3 + 2700924*n**4 - 803066*n**5 + 162507*n**6 - 22400*n**7 + 2046*n**8 - 114*n**9 + 3*n**10))/n**11 +
            D2pi2 * (2*(-3240000 + 9396000*n - 11092500*n**2 + 7121160*n**3 - 2764561*n**4 + 671649*n**5 - 99118*n**6 + 7365*n**7 + 56*n**8 - 54*n**9 + 3*n**10))/n**11 +
            Dpi2z * ((-5 + n)**2*(5184 - 7776*n - 396*n**2 + 5940*n**3 - 3959*n**4 + 1166*n**5 - 169*n**6 + 10*n**7))/n**11 +
            pi3 * (3*(20 - 9*n + n**2)**2*(-36 + 60*n - 25*n**2 + n**4))/n**10 +
            D4s1 * (4*(702000 - 2178900*n + 2822250*n**2 - 2045670*n**3 + 931302*n**4 - 281395*n**5 + 57754*n**6 - 8035*n**7 + 733*n**8 - 40*n**9 + n**10))/n**11 +
            D3zs1 * (2*(-324000 + 1045800*n - 1413000*n**2 + 1064030*n**3 - 497214*n**4 + 150791*n**5 - 29910*n**6 + 3770*n**7 - 276*n**8 + 9*n**9))/n**11 +
            D2pis1 * (1944000 - 5950800*n + 7518600*n**2 - 5219580*n**3 + 2209138*n**4 - 591144*n**5 + 99070*n**6 - 9750*n**7 + 472*n**8 - 6*n**9)/n**11 +
            Dpizs1 * (2*(-3 + n)**2*(-1200 + 2060*n - 564*n**2 - 685*n**3 + 495*n**4 - 115*n**5 + 9*n**6))/n**11 +
            pi2s1 * (3*(-2 + n)**2*(900 - 1545*n + 778*n**2 - 136*n**3 + 2*n**4 + n**5))/n**10 +
            D2s2 * (-2*(55800 - 191610*n + 274975*n**2 - 217032*n**3 + 103576*n**4 - 30720*n**5 + 5545*n**6 - 558*n**7 + 24*n**8))/n**11 +
            Dzs2 * -(((-1 + n)**2*(-360 + 702*n - 521*n**2 + 184*n**3 - 31*n**4 + 2*n**5))/n**11.) +
            pis2 * (-3*(-1 + n)**2*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**10 +
            s3 * 0 +
            D4 * (2*(-304200 + 955500*n - 1254000*n**2 + 920082*n**3 - 422470*n**4 + 127796*n**5 - 25892*n**6 + 3455*n**7 - 282*n**8 + 11*n**9))/n**11 +
            D3z * (97200 - 329400*n + 469200*n**2 - 371512*n**3 + 180630*n**4 - 55751*n**5 + 10781*n**6 - 1209*n**7 + 61*n**8)/n**11 +
            D2pi * (-583200 + 1879200*n - 2525400*n**2 + 1876032*n**3 - 850604*n**4 + 242722*n**5 - 42959*n**6 + 4447*n**7 - 245*n**8 + 7*n**9)/n**11 +
            Dpiz * (3600 - 10320*n + 10936*n**2 - 4738*n**3 + 30*n**4 + 689*n**5 - 215*n**6 + 17*n**7 + n**8)/n**11 +
            pi2 * ((-2 + n)**2*(-675 + 1260*n - 723*n**2 + 147*n**3 - 10*n**4 + n**5))/n**10 +
            D2s1 * (2*(-2 + n)**2*(4185 - 10860*n + 10835*n**2 - 5401*n**3 + 1409*n**4 - 175*n**5 + 7*n**6))/n**11 +
            Dzs1 * ((-1 + n)**2*(-60 + 146*n - 138*n**2 + 62*n**3 - 13*n**4 + n**5))/n**11 +
            pis1 * ((-1 + n)**2*(90 - 129*n + 66*n**2 - 14*n**3 + n**4))/n**10 +
            s2 * 0 +
            D2 * (-1922 + 7626*n - 12760*n**2 + 11716*n**3 - 6395*n**4 + 2081*n**5 - 375*n**6 + 29*n**7)/n**11 +
            Dz * (-1 + n)**6/n**11 +
            pi * ((-1 + n)**3*(3 - 3*n + n**2))/n**10 +
            s1 * 0][0]

def adjust_D3piz(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (24*(14400 - 40080*n + 44880*n**2 - 27108*n**3 + 9874*n**4 - 2265*n**5 + 325*n**6 - 27*n**7 + n**8))/n**11 +
            D5z * (18*(-144000 + 415200*n - 487920*n**2 + 313528*n**3 - 123500*n**4 + 31394*n**5 - 5225*n**6 + 557*n**7 - 35*n**8 + n**9))/n**11 +
            D4pi * (36*(1440000 - 4296000*n + 5285760*n**2 - 3600112*n**3 + 1524296*n**4 - 424198*n**5 + 79509*n**6 - 10048*n**7 + 834*n**8 - 42*n**9 + n**10))/n**11 +
            D3piz * (-17280000 + 52992000*n - 67684800*n**2 + 48364320*n**3 - 21738000*n**4 + 6509708*n**5 - 1335012*n**6 + 188949*n**7 - 18300*n**8 + 1182*n**9 - 48*n**10 + n**11)/n**11 +
            D2pi2 * (12*(-5 + n)**2*(86400 - 230400*n + 243672*n**2 - 137508*n**3 + 46526*n**4 - 9925*n**5 + 1338*n**6 - 107*n**7 + 4*n**8))/n**11 +
            Dpi2z * (3*(20 - 9*n + n**2)**2*(-432 + 720*n - 264*n**2 - 72*n**3 + 59*n**4 - 12*n**5 + n**6))/n**11 +
            pi3 * (16*(2 - 3*n + n**2)*(-60 + 47*n - 12*n**2 + n**3)**2)/n**10 +
            D4s1 * (-6*(1872000 - 5642400*n + 7022400*n**2 - 4833720*n**3 + 2061452*n**4 - 574212*n**5 + 106629*n**6 - 13140*n**7 + 1038*n**8 - 48*n**9 + n**10))/n**11 +
            D3zs1 * (3*(864000 - 2764800*n + 3699600*n**2 - 2760160*n**3 + 1281264*n**4 - 388716*n**5 + 78409*n**6 - 10440*n**7 + 886*n**8 - 44*n**9 + n**10))/n**11 +
            D2pis1 * (-12*(648000 - 2073600*n + 2789100*n**2 - 2109900*n**3 + 1004693*n**4 - 316965*n**5 + 67419*n**6 - 9570*n**7 + 867*n**8 - 45*n**9 + n**10))/n**11 +
            Dpizs1 * (3*(-2 + n)**2*(7200 - 14160*n + 9254*n**2 - 2477*n**3 + 122*n**4 + 76*n**5 - 16*n**6 + n**7))/n**11 +
            pi2s1 * (-4*(6 - 5*n + n**2)**2*(200 - 310*n + 129*n**2 - 20*n**3 + n**4))/n**10 +
            D2s2 * (-6*(-2 + n)**2*(-18600 + 46670*n - 45425*n**2 + 23002*n**3 - 6672*n**4 + 1124*n**5 - 103*n**6 + 4*n**7))/n**11 +
            Dzs2 * (-3*(-2 + n)**3*(-1 + n)**2*(-60 + 47*n - 12*n**2 + n**3))/n**11 +
            pis2 * (-4*(2 - 3*n + n**2)**2*(-60 + 47*n - 12*n**2 + n**3))/n**10 +
            s3 * 0 +
            D4 * (2433600 - 7410000*n + 9327240*n**2 - 6487824*n**3 + 2787140*n**4 - 777210*n**5 + 142983*n**6 - 17160*n**7 + 1284*n**8 - 54*n**9 + n**10)/n**11 +
            D3z * (3*(-129600 + 432000*n - 604140*n**2 + 469814*n**3 - 225234*n**4 + 69333*n**5 - 13767*n**6 + 1713*n**7 - 123*n**8 + 4*n**9))/n**11 +
            D2pi * (3*(777600 - 2592000*n + 3651240*n**2 - 2893444*n**3 + 1435496*n**4 - 466080*n**5 + 99912*n**6 - 13839*n**7 + 1167*n**8 - 53*n**9 + n**10))/n**11 +
            Dpiz * ((-2 + n)**2*(-3600 + 8520*n - 7531*n**2 + 3320*n**3 - 823*n**4 + 127*n**5 - 14*n**6 + n**7))/n**11 +
            pi2 * ((-2 + n)**3*(-1 + n)*(30 - 13*n + n**2)**2)/n**10 +
            D2s1 * (6*(-2 + n)**2*(-5580 + 14745*n - 15251*n**2 + 8155*n**3 - 2443*n**4 + 407*n**5 - 34*n**6 + n**7))/n**11 +
            Dzs1 * ((2 - 3*n + n**2)**2*(60 - 101*n + 60*n**2 - 14*n**3 + n**4))/n**11 +
            pis1 * ((-2 + n)**3*(-1 + n)**2*(30 - 13*n + n**2))/n**10 +
            s2 * 0 +
            D2 * ((-2 + n)**3*(-961 + 2356*n - 2119*n**2 + 883*n**3 - 172*n**4 + 13*n**5))/n**11 +
            Dz * ((-2 + n)**2*(-1 + n)**5)/n**11 +
            pi * (2 - 3*n + n**2)**3/n**10 +
            s1 * 0][0]

def adjust_D2pi2(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (24*(-3600 + 9420*n - 9610*n**2 + 5074*n**3 - 1525*n**4 + 265*n**5 - 25*n**6 + n**7))/n**11 +
            D5z * (24*(-5 + n)**2*(1080 - 2538*n + 2199*n**2 - 920*n**3 + 200*n**4 - 22*n**5 + n**6))/n**11 +
            D4pi * (24*(-540000 + 1566000*n - 1850370*n**2 + 1192159*n**3 - 467991*n**4 + 117382*n**5 - 19005*n**6 + 1936*n**7 - 114*n**8 + 3*n**9))/n**11 +
            D3piz * (4*(-5 + n)**2*(43200 - 115200*n + 121068*n**2 - 66902*n**3 + 21594*n**4 - 4225*n**5 + 497*n**6 - 33*n**7 + n**8))/n**11 +
            D2pi2 * ((20 - 9*n + n**2)**2*(-16200 + 37800*n - 32742*n**2 + 14040*n**3 - 3301*n**4 + 431*n**5 - 29*n**6 + n**7))/n**11 +
            Dpi2z * ((-60 + 47*n - 12*n**2 + n**3)**2*(36 - 60*n + 29*n**2 - 6*n**3 + n**4))/n**11 +
            pi3 * ((-1 + n)*(120 - 154*n + 71*n**2 - 14*n**3 + n**4)**2)/n**10 +
            D4s1 * (-12*(-234000 + 684300*n - 815450*n**2 + 528450*n**3 - 207417*n**4 + 51499*n**5 - 8130*n**6 + 790*n**7 - 43*n**8 + n**9))/n**11 +
            D3zs1 * (2*(-324000 + 1027800*n - 1354500*n**2 + 985390*n**3 - 439577*n**4 + 125445*n**5 - 23030*n**6 + 2640*n**7 - 173*n**8 + 5*n**9))/n**11 +
            D2pis1 * (1944000 - 6490800*n + 9169200*n**2 - 7305000*n**3 + 3661908*n**4 - 1212347*n**5 + 269165*n**6 - 39650*n**7 + 3722*n**8 - 203*n**9 + 5*n**10)/n**11 +
            Dpizs1 * ((-3 + n)**2*(-2400 + 6920*n - 7988*n**2 + 4910*n**3 - 1799*n**4 + 407*n**5 - 53*n**6 + 3*n**7))/n**11 +
            pi2s1 * ((6 - 5*n + n**2)**2*(100 - 205*n + 137*n**2 - 35*n**3 + 3*n**4))/n**10 +
            D2s2 * (2*(-55800 + 200010*n - 301955*n**2 + 252872*n**3 - 129597*n**4 + 42118*n**5 - 8655*n**6 + 1078*n**7 - 73*n**8 + 2*n**9))/n**11 +
            Dzs2 * ((-3 + n)**2*(-1 + n)**3*(-40 + 38*n - 11*n**2 + n**3))/n**11 +
            pis2 * ((-1 + n)**3*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**10 +
            s3 * 0 +
            D4 * (2*(-304200 + 897000*n - 1077960*n**2 + 702695*n**3 - 275808*n**4 + 67775*n**5 - 10416*n**6 + 961*n**7 - 48*n**8 + n**9))/n**11 +
            D3z * ((-3 + n)**2*(10800 - 28200*n + 28190*n**2 - 14005*n**3 + 3679*n**4 - 491*n**5 + 27*n**6))/n**11 +
            D2pi * (-583200 + 2008800*n - 2927460*n**2 + 2391940*n**3 - 1214285*n**4 + 398433*n**5 - 84662*n**6 + 11262*n**7 - 857*n**8 + 29*n**9)/n**11 +
            Dpiz * ((15 - 14*n + 3*n**2)**2*(16 - 32*n + 22*n**2 - 7*n**3 + n**4))/n**11 +
            pi2 * ((-1 + n)*(30 - 43*n + 20*n**2 - 3*n**3)**2)/n**10 +
            D2s1 * (2*(16740 - 61770*n + 96046*n**2 - 82511*n**3 + 42907*n**4 - 13830*n**5 + 2694*n**6 - 289*n**7 + 13*n**8))/n**11 +
            Dzs1 * ((-2 + n)**2*(-1 + n)**3*(15 - 14*n + 3*n**2))/n**11 +
            pis1 * ((-1 + n)**3*(-30 + 43*n - 20*n**2 + 3*n**3))/n**10 +
            s2 * 0 +
            D2 * (-1922 + 7564*n - 12511*n**2 + 11309*n**3 - 6048*n**4 + 1918*n**5 - 335*n**6 + 25*n**7)/n**11 +
            Dz * (-1 + n)**6/n**11 +
            pi * (-1 + n)**5/n**10 +
            s1 * 0][0]

def adjust_Dpi2z(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (480*(720 - 1764*n + 1624*n**2 - 735*n**3 + 175*n**4 - 21*n**5 + n**6))/n**11 +
            D5z * (600*(-6 + n)**2*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5))/n**11 +
            D4pi * (2400*(30 - 11*n + n**2)**2*(24 - 50*n + 35*n**2 - 10*n**3 + n**4))/n**11 +
            D3piz * (200*(-120 + 74*n - 15*n**2 + n**3)**2*(-6 + 11*n - 6*n**2 + n**3))/n**11 +
            D2pi2 * (100*(2 - 3*n + n**2)*(360 - 342*n + 119*n**2 - 18*n**3 + n**4)**2)/n**11 +
            Dpi2z * ((-1 + n)*(-720 + 1044*n - 580*n**2 + 155*n**3 - 20*n**4 + n**5)**2)/n**11 +
            pi3 * 0 +
            D4s1 * (-240*(46800 - 132660*n + 151100*n**2 - 91903*n**3 + 32998*n**4 - 7210*n**5 + 940*n**6 - 67*n**7 + 2*n**8))/n**11 +
            D3zs1 * (-20*(-4 + n)**2*(-8100 + 21420*n - 21915*n**2 + 11351*n**3 - 3212*n**4 + 492*n**5 - 37*n**6 + n**7))/n**11 +
            D2pis1 * (-20*(12 - 7*n + n**2)**2*(2700 - 6240*n + 5225*n**2 - 2042*n**3 + 390*n**4 - 34*n**5 + n**6))/n**11 +
            Dpizs1 * ((-24 + 26*n - 9*n**2 + n**3)**2*(150 - 295*n + 183*n**2 - 41*n**3 + 3*n**4))/n**11 +
            pi2s1 * 0 +
            D2s2 * (-10*(-2 + n)**2*(-11160 + 29682*n - 30815*n**2 + 16495*n**3 - 4986*n**4 + 860*n**5 - 79*n**6 + 3*n**7))/n**11 +
            Dzs2 * ((-2 + n)**2*(-1 + n)**3*(360 - 342*n + 119*n**2 - 18*n**3 + n**4))/n**11 +
            pis2 * 0 +
            s3 * 0 +
            D4 * (24*(65 - 25*n + 2*n**2)**2*(24 - 50*n + 35*n**2 - 10*n**3 + n**4))/n**11 +
            D3z * (2*(-180 + 125*n - 24*n**2 + n**3)**2*(-6 + 11*n - 6*n**2 + n**3))/n**11 +
            D2pi * (4*(2 - 3*n + n**2)*(540 - 555*n + 197*n**2 - 27*n**3 + n**4)**2)/n**11 +
            Dpiz * ((-1 + n)*(120 - 202*n + 123*n**2 - 32*n**3 + 3*n**4)**2)/n**11 +
            pi2 * 0 +
            D2s1 * (2*(-2 + n)**2*(-16740 + 45825*n - 49022*n**2 + 26773*n**3 - 8052*n**4 + 1319*n**5 - 106*n**6 + 3*n**7))/n**11 +
            Dzs1 * ((-2 + n)**2*(-1 + n)**3*(-60 + 71*n - 26*n**2 + 3*n**3))/n**11 +
            pis1 * 0 +
            s2 * 0 +
            D2 * ((-2 + n)**3*(-1 + n)*(31 - 22*n + 3*n**2)**2)/n**11 +
            Dz * ((-2 + n)**2*(-1 + n)**5)/n**11 +
            pi * 0 +
            s1 * 0][0]

def adjust_pi3(n, stat_names, stats):
    D6 = stats[stat_names.index('D^6')]
    D5z = stats[stat_names.index('D^5_z')]
    D4pi = stats[stat_names.index('D^4_pi^1')]
    D3piz = stats[stat_names.index('D^3_pi^1_z')]
    D2pi2 = stats[stat_names.index('D^2_pi^2')]
    Dpi2z = stats[stat_names.index('D^1_pi^2_z')]
    pi3 = stats[stat_names.index('pi^3')]
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D6 * (720*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5))/n**11 +
            D5z * (1080*(-5 + n)**2*(24 - 50*n + 35*n**2 - 10*n**3 + n**4))/n**11 +
            D4pi * (5400*(20 - 9*n + n**2)**2*(-6 + 11*n - 6*n**2 + n**3))/n**11 +
            D3piz * (600*(2 - 3*n + n**2)*(-60 + 47*n - 12*n**2 + n**3)**2)/n**11 +
            D2pi2 * (450*(-1 + n)*(120 - 154*n + 71*n**2 - 14*n**3 + n**4)**2)/n**11 +
            Dpi2z * (9*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5)**2)/n**11 +
            pi3 * (-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5)**2/n**10 +
            D4s1 * (-360*(-7800 + 21410*n - 23205*n**2 + 13097*n**3 - 4200*n**4 + 770*n**5 - 75*n**6 + 3*n**7))/n**11 +
            D3zs1 * (-60*(-3 + n)**2*(1200 - 2940*n + 2680*n**2 - 1173*n**3 + 259*n**4 - 27*n**5 + n**6))/n**11 +
            D2pis1 * (-90*(6 - 5*n + n**2)**2*(-600 + 1170*n - 755*n**2 + 209*n**3 - 25*n**4 + n**5))/n**11 +
            Dpizs1 * (6*(-6 + 11*n - 6*n**2 + n**3)**2*(-100 + 105*n - 32*n**2 + 3*n**3))/n**11 +
            pi2s1 * ((-6 + 11*n - 6*n**2 + n**3)**2*(-100 + 105*n - 32*n**2 + 3*n**3))/n**10 +
            D2s2 * (-30*(3720 - 13894*n + 21825*n**2 - 18874*n**3 + 9837*n**4 - 3166*n**5 + 615*n**6 - 66*n**7 + 3*n**8))/n**11 +
            Dzs2 * (3*(-1 + n)**4*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**11 +
            pis2 * ((-1 + n)**4*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**10 +
            s3 * 0 +
            D4 * (24*(65 - 30*n + 3*n**2)**2*(-6 + 11*n - 6*n**2 + n**3))/n**11 +
            D3z * (6*(2 - 3*n + n**2)*(-90 + 75*n - 18*n**2 + n**3)**2)/n**11 +
            D2pi * (18*(-1 + n)*(180 - 240*n + 111*n**2 - 20*n**3 + n**4)**2)/n**11 +
            Dpiz * (4*(30 - 73*n + 63*n**2 - 23*n**3 + 3*n**4)**2)/n**11 +
            pi2 * (30 - 73*n + 63*n**2 - 23*n**3 + 3*n**4)**2/n**10 +
            D2s1 * (6*(5580 - 21120*n + 33561*n**2 - 29216*n**3 + 15177*n**4 - 4780*n**5 + 879*n**6 - 84*n**7 + 3*n**8))/n**11 +
            Dzs1 * (2*(-1 + n)**4*(-30 + 43*n - 20*n**2 + 3*n**3))/n**11 +
            pis1 * ((-1 + n)**4*(-30 + 43*n - 20*n**2 + 3*n**3))/n**10 +
            s2 * 0 +
            D2 * (2*(-1 + n)*(31 - 45*n + 21*n**2 - 3*n**3)**2)/n**11 +
            Dz * (-1 + n)**6/n**11 +
            pi * (-1 + n)**6/n**10 +
            s1 * 0][0]

def adjust_D4s1(n, stat_names, stats):
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D4s1 * (-43200 + 127440*n - 155400*n**2 + 105768*n**3 - 45408*n**4 + 13054*n**5 - 2565*n**6 + 337*n**7 - 27*n**8 + n**9)/n**9 +
            D3zs1 * (6*(7200 - 21240*n + 25660*n**2 - 16960*n**3 + 6824*n**4 - 1735*n**5 + 275*n**6 - 25*n**7 + n**8))/n**9 +
            D2pis1 * (6*(-21600 + 60120*n - 66360*n**2 + 38170*n**3 - 12326*n**4 + 2155*n**5 - 155*n**6 - 5*n**7 + n**8))/n**9 +
            Dpizs1 * (2*(-3 + n)**2*(240 - 268*n - 96*n**2 + 173*n**3 - 54*n**4 + 5*n**5))/n**9 +
            pi2s1 * (3*(-720 + 1524*n - 1076*n**2 + 285*n**3 - 5*n**4 - 9*n**5 + n**6))/n**8 +
            D2s2 * (25200 - 74340*n + 88490*n**2 - 55926*n**3 + 20450*n**4 - 4350*n**5 + 500*n**6 - 24*n**7)/n**9 +
            Dzs2 * (-360 + 1062*n - 1223*n**2 + 705*n**3 - 215*n**4 + 33*n**5 - 2*n**6)/n**9 +
            pis2 * (-3*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5))/n**8 +
            s3 * 0 +
            D4 * (8*(2340 - 6990*n + 8640*n**2 - 5949*n**3 + 2570*n**4 - 736*n**5 + 141*n**6 - 17*n**7 + n**8))/n**9 +
            D3z * (12*(-1080 + 3360*n - 4300*n**2 + 2993*n**3 - 1246*n**4 + 316*n**5 - 46*n**6 + 3*n**7))/n**9 +
            D2pi * (-12*(-6480 + 19080*n - 22560*n**2 + 13998*n**3 - 4889*n**4 + 929*n**5 - 79*n**6 + n**7))/n**9 +
            Dpiz * (4*(-360 + 816*n - 532*n**2 - 47*n**3 + 187*n**4 - 73*n**5 + 9*n**6))/n**9 +
            pi2 * (6*(-2 + n)**2*(45 - 57*n + 11*n**2 + n**3))/n**8 +
            D2s1 * (-9792 + 30924*n - 39730*n**2 + 27030*n**3 - 10450*n**4 + 2243*n**5 - 232*n**6 + 7*n**7)/n**9 +
            Dzs1 * (72 - 252*n + 352*n**2 - 248*n**3 + 91*n**4 - 16*n**5 + n**6)/n**9 +
            pis1 * (-108 + 270*n - 246*n**2 + 101*n**3 - 18*n**4 + n**5)/n**8 +
            s2 * 0 +
            D2 * (4*(217 - 756*n + 1076*n**2 - 806*n**3 + 337*n**4 - 75*n**5 + 7*n**6))/n**9 +
            Dz * (2*(-1 + n)**5)/n**9 +
            pi * (2*(-1 + n)**2*(3 - 3*n + n**2))/n**8 +
            s1 * 0][0]

def adjust_D3zs1(n, stat_names, stats):
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D4s1 * (12*(14400 - 40080*n + 44880*n**2 - 27108*n**3 + 9874*n**4 - 2265*n**5 + 325*n**6 - 27*n**7 + n**8))/n**9 +
            D3zs1 * (-172800 + 495360*n - 575760*n**2 + 362880*n**3 - 138552*n**4 + 33664*n**5 - 5295*n**6 + 535*n**7 - 33*n**8 + n**9)/n**9 +
            D2pis1 * (12*(43200 - 123840*n + 143940*n**2 - 90660*n**3 + 34471*n**4 - 8235*n**5 + 1225*n**6 - 105*n**7 + 4*n**8))/n**9 +
            Dpizs1 * (3*(-5760 + 13632*n - 11416*n**2 + 3764*n**3 + 114*n**4 - 457*n**5 + 141*n**6 - 19*n**7 + n**8))/n**9 +
            pi2s1 * (16*(-3 + n)**2*(40 - 78*n + 49*n**2 - 12*n**3 + n**4))/n**8 +
            D2s2 * (-6*(16800 - 50360*n + 61660*n**2 - 40942*n**3 + 16323*n**4 - 4045*n**5 + 615*n**6 - 53*n**7 + 2*n**8))/n**9 +
            Dzs2 * (-3*(-2 + n)**3*(60 - 107*n + 59*n**2 - 13*n**3 + n**4))/n**9 +
            pis2 * (-4*(-2 + n)**2*(60 - 107*n + 59*n**2 - 13*n**3 + n**4))/n**8 +
            s3 * 0 +
            D4 * (-4*(18720 - 52680*n + 59724*n**2 - 36468*n**3 + 13360*n**4 - 3051*n**5 + 427*n**6 - 33*n**7 + n**8))/n**9 +
            D3z * (6*(8640 - 25920*n + 31668*n**2 - 20866*n**3 + 8190*n**4 - 1981*n**5 + 293*n**6 - 25*n**7 + n**8))/n**9 +
            D2pi * (-24*(12960 - 38880*n + 47622*n**2 - 31567*n**3 + 12490*n**4 - 3031*n**5 + 439*n**6 - 34*n**7 + n**8))/n**9 +
            Dpiz * (2*(-2 + n)**2*(720 - 1272*n + 671*n**2 - 115*n**3 - 7*n**4 + 3*n**5))/n**9 +
            pi2 * (-8*(10 - 11*n + n**2)*(6 - 5*n + n**2)**2)/n**8 +
            D2s1 * (3*(13056 - 41408*n + 53944*n**2 - 37960*n**3 + 15790*n**4 - 3954*n**5 + 573*n**6 - 42*n**7 + n**8))/n**9 +
            Dzs1 * ((-2 + n)**2*(-72 + 198*n - 202*n**2 + 93*n**3 - 18*n**4 + n**5))/n**9 +
            pis1 * ((-2 + n)**2*(72 - 144*n + 91*n**2 - 20*n**3 + n**4))/n**8 +
            s2 * 0 +
            D2 * (2*(-2 + n)**3*(217 - 421*n + 263*n**2 - 65*n**3 + 6*n**4))/n**9 +
            Dz * (2*(-2 + n)**2*(-1 + n)**4)/n**9 +
            pi * (2*(-2 + n)**3*(-1 + n)**2)/n**8 +
            s1 * 0][0]

def adjust_D2pis1(n, stat_names, stats):
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D4s1 * (12*(-3600 + 9420*n - 9610*n**2 + 5074*n**3 - 1525*n**4 + 265*n**5 - 25*n**6 + n**7))/n**9 +
            D3zs1 * (2*(21600 - 60120*n + 66960*n**2 - 39780*n**3 + 13999*n**4 - 3030*n**5 + 400*n**6 - 30*n**7 + n**8))/n**9 +
            D2pis1 * (-129600 + 382320*n - 461880*n**2 + 305400*n**3 - 123226*n**4 + 31729*n**5 - 5260*n**6 + 550*n**7 - 34*n**8 + n**9)/n**9 +
            Dpizs1 * ((-3 + n)**2*(480 - 1096*n + 940*n**2 - 418*n**3 + 109*n**4 - 16*n**5 + n**6))/n**9 +
            pi2s1 * ((6 - 5*n + n**2)**2*(-20 + 29*n - 10*n**2 + n**3))/n**8 +
            D2s2 * (2*(12600 - 38370*n + 47585*n**2 - 31703*n**3 + 12474*n**4 - 2975*n**5 + 420*n**6 - 32*n**7 + n**8))/n**9 +
            Dzs2 * ((3 - 4*n + n**2)**2*(-40 + 38*n - 11*n**2 + n**3))/n**9 +
            pis2 * ((-1 + n)**2*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**8 +
            s3 * 0 +
            D4 * (-4*(-4680 + 12360*n - 12732*n**2 + 6761*n**3 - 2022*n**4 + 342*n**5 - 30*n**6 + n**7))/n**9 +
            D3z * (2*(-3 + n)**2*(-720 + 1600*n - 1258*n**2 + 447*n**3 - 74*n**4 + 5*n**5))/n**9 +
            D2pi * (77760 - 237600*n + 297384*n**2 - 201728*n**3 + 81718*n**4 - 20348*n**5 + 3064*n**6 - 260*n**7 + 10*n**8)/n**9 +
            Dpiz * (2*(-3 + n)**2*(-80 + 208*n - 206*n**2 + 101*n**3 - 26*n**4 + 3*n**5))/n**9 +
            pi2 * (2*(6 - 5*n + n**2)**2*(5 - 8*n + 3*n**2))/n**8 +
            D2s1 * (2*(-4896 + 15594*n - 20251*n**2 + 14001*n**3 - 5583*n**4 + 1285*n**5 - 158*n**6 + 8*n**7))/n**9 +
            Dzs1 * (2*(2 - 3*n + n**2)**2*(9 - 9*n + 2*n**2))/n**9 +
            pis1 * (2*(-1 + n)**2*(-18 + 27*n - 13*n**2 + 2*n**3))/n**8 +
            s2 * 0 +
            D2 * (868 - 2948*n + 4066*n**2 - 2932*n**3 + 1172*n**4 - 248*n**5 + 22*n**6)/n**9 +
            Dz * (2*(-1 + n)**5)/n**9 +
            pi * (2*(-1 + n)**4)/n**8 +
            s1 * 0][0]

def adjust_Dpizs1(n, stat_names, stats):
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D4s1 * (240*(720 - 1764*n + 1624*n**2 - 735*n**3 + 175*n**4 - 21*n**5 + n**6))/n**9 +
            D3zs1 * (60*(-4 + n)**2*(-180 + 396*n - 307*n**2 + 107*n**3 - 17*n**4 + n**5))/n**9 +
            D2pis1 * (60*(12 - 7*n + n**2)**2*(60 - 112*n + 65*n**2 - 14*n**3 + n**4))/n**9 +
            Dpizs1 * ((-30 + 41*n - 12*n**2 + n**3)*(-24 + 26*n - 9*n**2 + n**3)**2)/n**9 +
            pi2s1 * 0 +
            D2s2 * (-10*(-2 + n)**2*(2520 - 5274*n + 3929*n**2 - 1420*n**3 + 270*n**4 - 26*n**5 + n**6))/n**9 +
            Dzs2 * ((2 - 3*n + n**2)**2*(360 - 342*n + 119*n**2 - 18*n**3 + n**4))/n**9 +
            pis2 * 0 +
            s3 * 0 +
            D4 * (-48*(1560 - 3850*n + 3573*n**2 - 1625*n**3 + 385*n**4 - 45*n**5 + 2*n**6))/n**9 +
            D3z * (-12*(-4 + n)**2*(-270 + 615*n - 496*n**2 + 176*n**3 - 26*n**4 + n**5))/n**9 +
            D2pi * (-24*(12 - 7*n + n**2)**2*(90 - 175*n + 107*n**2 - 23*n**3 + n**4))/n**9 +
            Dpiz * (2*(5 - 8*n + 3*n**2)*(-24 + 26*n - 9*n**2 + n**3)**2)/n**9 +
            pi2 * 0 +
            D2s1 * (2*(-2 + n)**2*(4896 - 10764*n + 8465*n**2 - 3112*n**3 + 558*n**4 - 44*n**5 + n**6))/n**9 +
            Dzs1 * (2*(2 - 3*n + n**2)**2*(-36 + 45*n - 17*n**2 + 2*n**3))/n**9 +
            pis1 * 0 +
            s2 * 0 +
            D2 * (2*(-2 + n)**3*(217 - 402*n + 228*n**2 - 46*n**3 + 3*n**4))/n**9 +
            Dz * (2*(-2 + n)**2*(-1 + n)**4)/n**9 +
            pi * 0 +
            s1 * 0][0]

def adjust_pi2s1(n, stat_names, stats):
    D4s1 = stats[stat_names.index('D^4_s1')]
    D3zs1 = stats[stat_names.index('D^3_z_s1')]
    D2pis1 = stats[stat_names.index('D^2_pi^1_s1')]
    Dpizs1 = stats[stat_names.index('D^1_pi^1_z_s1')]
    pi2s1 = stats[stat_names.index('pi^2_s1')]
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D4s1 * (360*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5))/n**9 +
            D3zs1 * (120*(-3 + n)**2*(40 - 78*n + 49*n**2 - 12*n**3 + n**4))/n**9 +
            D2pis1 * (180*(6 - 5*n + n**2)**2*(-20 + 29*n - 10*n**2 + n**3))/n**9 +
            Dpizs1 * (6*(20 - 9*n + n**2)*(-6 + 11*n - 6*n**2 + n**3)**2)/n**9 +
            pi2s1 * ((20 - 9*n + n**2)*(-6 + 11*n - 6*n**2 + n**3)**2)/n**8 +
            D2s2 * (-30*(-840 + 2638*n - 3339*n**2 + 2219*n**3 - 840*n**4 + 182*n**5 - 21*n**6 + n**7))/n**9 +
            Dzs2 * (3*(-1 + n)**3*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**9 +
            pis2 * ((-1 + n)**3*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**8 +
            s3 * 0 +
            D4 * (-48*(-390 + 895*n - 738*n**2 + 278*n**3 - 48*n**4 + 3*n**5))/n**9 +
            D3z * (-24*(-3 + n)**2*(60 - 120*n + 77*n**2 - 18*n**3 + n**4))/n**9 +
            D2pi * (-72*(6 - 5*n + n**2)**2*(-30 + 45*n - 16*n**2 + n**3))/n**9 +
            Dpiz * (8*(-5 + 3*n)*(-6 + 11*n - 6*n**2 + n**3)**2)/n**9 +
            pi2 * (2*(-5 + 3*n)*(-6 + 11*n - 6*n**2 + n**3)**2)/n**8 +
            D2s1 * (6*(-1632 + 5242*n - 6771*n**2 + 4543*n**3 - 1692*n**4 + 342*n**5 - 33*n**6 + n**7))/n**9 +
            Dzs1 * (4*(-1 + n)**3*(-18 + 27*n - 13*n**2 + 2*n**3))/n**9 +
            pis1 * (2*(-1 + n)**3*(-18 + 27*n - 13*n**2 + 2*n**3))/n**8 +
            s2 * 0 +
            D2 * (4*(217 - 718*n + 949*n**2 - 640*n**3 + 231*n**4 - 42*n**5 + 3*n**6))/n**9 +
            Dz * (2*(-1 + n)**5)/n**9 +
            pi * (2*(-1 + n)**5)/n**8 +
            s1 * 0][0]

def adjust_D2s2(n, stat_names, stats):
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    s3 = stats[stat_names.index('1_s3')]
    D4 = stats[stat_names.index('D^4')]
    D3z = stats[stat_names.index('D^3_z')]
    D2pi = stats[stat_names.index('D^2_pi^1')] 
    Dpiz = stats[stat_names.index('D^1_pi^1_z')]
    pi2 = stats[stat_names.index('pi^2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    s2 = stats[stat_names.index('1_s2')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    s1 = stats[stat_names.index('1_s1')]
    return [D2s2 * (-3600 + 9420*n - 9610*n**2 + 5074*n**3 - 1525*n**4 + 265*n**5 - 25*n**6 + n**7)/n**7 +
            Dzs2 * ((-3 + n)**2*(40 - 78*n + 49*n**2 - 12*n**3 + n**4))/n**7 +
            pis2 * (-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5)/n**6 +
            s3 * 0 +
            D4 * 0 +
            D3z * 0 +
            D2pi * 0 +
            Dpiz * 0 +
            pi2 * 0 +
            D2s1 * (1080 - 2940*n + 3122*n**2 - 1687*n**3 + 497*n**4 - 77*n**5 + 5*n**6)/n**7 +
            Dzs1 * ((-2 + n)**2*(-15 + 29*n - 17*n**2 + 3*n**3))/n**7 +
            pis1 * (30 - 73*n + 63*n**2 - 23*n**3 + 3*n**4)/n**6 +
            s2 * 0 +
            D2 * (4*(-31 + 92*n - 106*n**2 + 60*n**3 - 17*n**4 + 2*n**5))/n**7 +
            Dz * (2*(-1 + n)**4)/n**7 +
            pi * (2*(-1 + n)**3)/n**6 +
            s1 * 0][0]

def adjust_Dzs2(n, stat_names, stats):
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    return [D2s2 * (20*(720 - 1764*n + 1624*n**2 - 735*n**3 + 175*n**4 - 21*n**5 + n**6))/n**7 +
            Dzs2 * ((-2 + n)**2*(-360 + 702*n - 461*n**2 + 137*n**3 - 19*n**4 + n**5))/n**7 +
            D2s1 * (-4*(1080 - 2730*n + 2599*n**2 - 1200*n**3 + 280*n**4 - 30*n**5 + n**6))/n**7 +
            Dzs1 * ((-2 + n)**2*(60 - 131*n + 97*n**2 - 29*n**3 + 3*n**4))/n**7 +
            D2 * (-4*(-2 + n)**2*(-31 + 53*n - 25*n**2 + 3*n**3))/n**7 +
            Dz * (2*(-2 + n)**2*(-1 + n)**3)/n**7][0]

def adjust_pis2(n, stat_names, stats):
    D2s2 = stats[stat_names.index('D^2_s2')]
    Dzs2 = stats[stat_names.index('D^1_z_s2')]
    pis2 = stats[stat_names.index('pi^1_s2')]
    D2s1 = stats[stat_names.index('D^2_s1')]
    Dzs1 = stats[stat_names.index('D^1_z_s1')]
    pis1 = stats[stat_names.index('pi^1_s1')]
    D2 = stats[stat_names.index('D^2')]
    Dz = stats[stat_names.index('D^1_z')]
    pi = stats[stat_names.index('pi^1')]
    return [D2s2 * (30*(-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5))/n**7 +
            Dzs2 * (3*(-1 + n)**2*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**7 +
            pis2 * ((-1 + n)**2*(120 - 154*n + 71*n**2 - 14*n**3 + n**4))/n**6 +
            D2s1 * (-6*(-180 + 420*n - 351*n**2 + 131*n**3 - 21*n**4 + n**5))/n**7 +
            Dzs1 * (2*(-1 + n)**2*(-30 + 43*n - 20*n**2 + 3*n**3))/n**7 +
            pis1 * ((-1 + n)**2*(-30 + 43*n - 20*n**2 + 3*n**3))/n**6 +
            D2 * (-4*(31 - 76*n + 66*n**2 - 24*n**3 + 3*n**4))/n**7 +
            Dz * (2*(-1 + n)**4)/n**7 +
            pi * (2*(-1 + n)**4)/n**6][0]

def adjust_s3(n, stat_names, stats):
    s3 = stats[stat_names.index('1_s3')]
    s2 = stats[stat_names.index('1_s2')]
    s1 = stats[stat_names.index('1_s1')]
    return [s3 * (-120 + 274*n - 225*n**2 + 85*n**3 - 15*n**4 + n**5)/n**5 +
            s2 * (30 - 73*n + 63*n**2 - 23*n**3 + 3*n**4)/n**5 +
            s1 * (-1 + n)**3/n**5][0]

"""
Correction for genotypes (with and without sampling)
"""

def order2correction_genotypes(stats, n):
    if n is None:
        # corrections for genotype without correcting for sampling bias
        return np.array([ 1./4, 1./2, 1., 1., 1.]) * np.array(stats)
    else:
        # corrections for genotype data with sampling bias
        return adjust_order2_sampling(n).dot(stats)

def order4correction_genotypes(stats, n):
    if n is None:
        # corrections for genotype without correcting for sampling bias
        return np.array([ 1./16, 1./8, 1./4, 1./2, 1., 1./4, 1./2, 1., 1., 1./4, 1./2, 1., 1., 1. ]) * np.array(stats)
    else:
        # corrections for genotype data with sampling bias
        return adjust_order4_sampling(n).dot(stats)

def adjust_order2_sampling(n):
    return np.array([[(-1. + 2*n - 2*n**2 + n**3)/(4.*n**3), (-1. + n)**2/(8.*n**3), (-1. + n)/(4.*n**2), 0, 0],
                     [(-1. + n)**2/n**3, (-1. + n)**3/(2.*n**3), 0, 0, 0],
                     [(-1. + 2*n)/(4.*n**3), (1. - 2*n)**2/(8.*n**3), (1. - 2*n)**2/(4.*n**2), 0, 0],
                     [0, 0, 0, 1 - 1./(2.*n), 0],
                     [0, 0, 0, 0, 1]])

def adjust_order4_sampling(n):
    return np.array([[ (-18 + 30*n - 18*n**2 + 12*n**3 - 11*n**4 + 8*n**5 - 4*n**6 + n**7)/(16.*n**7), (3*(18 - 30*n + 14*n**2 - 5*n**3 + 6*n**4 - 4*n**5 + n**6))/(16.*n**7), (3*(-54 + 72*n + 8*n**2 - 39*n**3 + 15*n**4 - 3*n**5 + n**6))/(8.*n**7), (18 + 12*n - 76*n**2 + 67*n**3 - 26*n**4 + 5*n**5)/(16.*n**7), (3*(-9 + 12*n - n**2 - 3*n**3 + n**4))/(16.*n**6), (-3*(-21 + 35*n - 9*n**2 - 7*n**3 + n**4 + n**5))/(16.*n**7), ((-1 + n)**2*(-3 - n + 2*n**2))/(32.*n**7), (3*(3 - 5*n + 2*n**2))/(32.*n**6), 0, ((-49 + 98*n - 52*n**2 + 6*n**3 - 10*n**4 + 7*n**5))/(64.*n**7), (-1 + n)**4/(128.*n**7), (-3 + 6*n - 4*n**2 + n**3)/(64.*n**6), 0, 0 ], ## D^4
                     [ (18 - 39*n + 39*n**2 - 30*n**3 + 17*n**4 - 6*n**5 + n**6)/(4.*n**7), (-108 + 252*n - 261*n**2 + 195*n**3 - 111*n**4 + 41*n**5 - 9*n**6 + n**7)/(8.*n**7), (3*(108 - 252*n + 241*n**2 - 153*n**3 + 79*n**4 - 27*n**5 + 4*n**6))/(4.*n**7), (-36 + 48*n + 29*n**2 - 84*n**3 + 61*n**4 - 21*n**5 + 3*n**6)/(8.*n**7), ((3 - 2*n)**2*(2 - 3*n + n**2))/(4.*n**6), (-3*(42 - 109*n + 118*n**2 - 81*n**3 + 42*n**4 - 14*n**5 + 2*n**6))/(8.*n**7), ((-1 + n)**2*(6 - 7*n + 2*n**2))/(16.*n**7), -((-1 + n)**2*(6 - 7*n + 2*n**2))/(8.*n**6), 0, ((49 - 140*n + 168*n**2 - 123*n**3 + 64*n**4 - 21*n**5 + 3*n**6))/(16.*n**7), (-1 + n)**5/(32.*n**7), ((-2 + n)*(-1 + n)**3)/(16.*n**6), 0, 0 ], ## D^3z
                     [ (-18 + 48*n - 55*n**2 + 36*n**3 - 13*n**4 + 2*n**5)/(16.*n**7), (108 - 324*n + 422*n**2 - 317*n**3 + 143*n**4 - 36*n**5 + 4*n**6)/(32.*n**7), (-324 + 1080*n - 1570*n**2 + 1329*n**3 - 708*n**4 + 233*n**5 - 44*n**6 + 4*n**7)/(16.*n**7), ((3 - 2*n)**2*(4 - 8*n + 6*n**2 - 3*n**3 + n**4))/(32.*n**7), ((3 - 2*n)**2*(-1 + n)**3)/(16.*n**6), (126 - 444*n + 661*n**2 - 547*n**3 + 268*n**4 - 72*n**5 + 8*n**6)/(32.*n**7), (-6 + 28*n - 49*n**2 + 43*n**3 - 20*n**4 + 4*n**5)/(64.*n**7), ((-1 + n)**2*(3 - 8*n + 4*n**2))/(32.*n**6), 0, ((-49 + 182*n - 278*n**2 + 225*n**3 - 100*n**4 + 20*n**5))/(64.*n**7), (1 - 3*n + 2*n**2)**2/(128.*n**7), ((1 - 2*n)**2*(-1 + n))/(64.*n**6), 0, 0 ], ## D^2pi
                     [ (3*(-1 + n)**2*(6 - 7*n + 2*n**2))/(4.*n**7), (9*(-3 + 2*n)*(2 - 3*n + n**2)**2)/(8.*n**7), (9*(6 - 13*n + 9*n**2 - 2*n**3)**2)/(4.*n**7), ((-1 + n)**3*(6 - 7*n + 2*n**2)**2)/(8.*n**7), 0, (-3*(-1 + n)**3*(-42 + 61*n - 28*n**2 + 4*n**3))/(8.*n**7), ((-1 + n)**3*(-6 + 19*n - 16*n**2 + 4*n**3))/(16.*n**7), 0, 0, ((7 - 2*n)**2*(-1 + n)**4)/(16.*n**7), ((1 - 2*n)**2*(-1 + n)**3)/(32.*n**7), 0, 0, 0 ], ## Dpiz
                     [ (3*(-3 + 11*n - 12*n**2 + 4*n**3))/(8.*n**7), (3*(3 - 2*n)**2*(1 - 3*n + 2*n**2))/(8.*n**7), (9*(-1 + 2*n)*(3 - 5*n + 2*n**2)**2)/(4.*n**7), (3 - 11*n + 12*n**2 - 4*n**3)**2/(8.*n**7), (3 - 11*n + 12*n**2 - 4*n**3)**2/(16.*n**6), (63 - 339*n + 684*n**2 - 648*n**3 + 288*n**4 - 48*n**5)/(16.*n**7), ((-1 + 2*n)**3*(3 - 5*n + 2*n**2))/(32.*n**7), ((-1 + 2*n)**3*(3 - 5*n + 2*n**2))/(32.*n**6), 0, ((-1 + 2*n)*(7 - 12*n + 4*n**2)**2)/(64.*n**7), (1 - 2*n)**4/(128.*n**7), (1 - 2*n)**4/(64.*n**6), 0, 0 ], ## pi^2
                     [ 0, 0, 0, 0, 0, (-18 + 48*n - 55*n**2 + 36*n**3 - 13*n**4 + 2*n**5)/(8.*n**5), (6 - 16*n + 17*n**2 - 9*n**3 + 2*n**4)/(16.*n**5), ((-1 + n)**2*(-3 + 2*n))/(8.*n**4), 0, ((-1 + n)**2*(7 - 6*n + 4*n**2))/(8.*n**5), ((-1 + n)**2*(-1 + 2*n))/(16.*n**5), (1 - 3*n + 2*n**2)/(8.*n**4), 0, 0 ], ## D^2s
                     [ 0, 0, 0, 0, 0, (3*(-1 + n)**2*(6 - 7*n + 2*n**2))/(2.*n**5), ((-1 + n)**3*(6 - 7*n + 2*n**2))/(4.*n**5), 0, 0, -((-1 + n)**3*(-7 + 2*n))/(2.*n**5), ((-1 + n)**3*(-1 + 2*n))/(4.*n**5), 0, 0, 0 ], ## Dzs
                     [ 0, 0, 0, 0, 0, (3*(-3 + 11*n - 12*n**2 + 4*n**3))/(4.*n**5), ((1 - 2*n)**2*(3 - 5*n + 2*n**2))/(8.*n**5), ((1 - 2*n)**2*(3 - 5*n + 2*n**2))/(8.*n**4), 0, -((-7 + 26*n - 28*n**2 + 8*n**3))/(8.*n**5), (-1 + 2*n)**3/(16.*n**5), (-1 + 2*n)**3/(8.*n**4), 0, 0 ], ## pis
                     [ 0, 0, 0, 0, 0, 0, 0, 0, (-3 + 11*n - 12*n**2 + 4*n**3)/(4.*n**3), 0, 0, 0, (1 - 2*n)**2/(8.*n**3), 0 ], ## s^2
                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1 + 2*n - 2*n**2 + n**3)/(4.*n**3), (-1 + n)**2/(8.*n**3), (-1 + n)/(4.*n**2), 0, 0 ], ## D^2
                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1 + n)**2/(1.*n**3), (-1 + n)**3/(2.*n**3), 0, 0, 0 ], ## Dz
                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, (-1 + 2*n)/(4.*n**3), (1 - 2*n)**2/(8.*n**3), (1 - 2*n)**2/(4.*n**2), 0, 0 ], ## pi
                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - 1/(2.*n), 0], ## s
                     [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]]) ## 1




def adjust_moment_genotype(name, stat_names, stats, sample_sizes):
    """
    Found using Mathematica and moment generating function for hypergeometric sampling
    """
    moment_type = name.split('_')[0]
    if moment_type == 'DD':
        popA = name.split('_')[1]
        popB = name.split('_')[2]
        if popA == popB:
            n1 = sample_sizes[int(popA)-1] # note that pop names are 1-indexed, not 0-indexed
            # need D_A^2, Dz_A, pi_A, which relies on (1-2p1)^2(1-2q1)^2, (1-2p1)^2, (1-2q1)^2, and 1
            mom1 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{0}'.format(popA))[0]]
            mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{0}'.format(popA))[0]]
            mom3 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{0}_{0}'.format(popA))[0]]
            mom4 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{0}'.format(popA))[0]]
            mom5 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{0}'.format(popA))[0]]
            return mom1 * (-1 + 2*n1 - 2*n1**2 + n1**3)/(4.*n1**3) + mom2 * (-1 + n1)**2/(8.*n1**3) + mom3 * (-1 + n1)/(64.*n1**2) - mom4 * (-1 + n1)/(64.*n1**2) - mom5 * (-1 + n1)/(64.*n1**2) + 1. * (-1 + n1)/(64.*n1**2)
        else:
            n1 = sample_sizes[int(popA)-1]
            n2 = sample_sizes[int(popB)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popA,popB))[0]]
            return mom1 * ((-1 + n1)*(-1 + n2))/(4.*n1*n2)
    elif moment_type == 'Dz':
        popD = name.split('_')[1]
        popp = name.split('_')[2]
        popq = name.split('_')[3]
        if popD == popp == popq:
            n1 = sample_sizes[int(popD)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{0}'.format(popD))[0]]
            return mom1 * (-1 + n1)**3/(2.*n1**3) + mom2 * (-1 + n1)**2/n1**3
        elif popD == popp or popD == popq:
            n1 = sample_sizes[int(popD)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            return mom1 * (-1 + n1)**2/(2.*n1**2)
        elif popp == popq:
            n1 = sample_sizes[int(popD)-1]
            n2 = sample_sizes[int(popp)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            try:
                mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popD,popp))[0]]
            except IndexError:
                mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popp,popD))[0]]
            return mom1 * (-1 + n1)/(2.*n1) + mom2 * (-1 + n1)/(n1*n2)
        else:
            n1 = sample_sizes[int(popD)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popD,popp,popq))[0]]
            return mom1 * (-1 + n1)/(2.*n1)
    elif moment_type == 'zz':
        popp1 = name.split('_')[1]
        popp2 = name.split('_')[2]
        popq1 = name.split('_')[3]
        popq2 = name.split('_')[4]
        if popp1 == popp2 == popq1 == popq2: # e.g. 1_1_1_1
            n1 = sample_sizes[int(popp1)-1]
            mom1 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{0}'.format(popp1))[0]]
            mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{0}'.format(popp1))[0]]
            mom3 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{0}_{0}'.format(popp1))[0]]
            mom4 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{0}'.format(popp1))[0]]
            mom5 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{0}'.format(popp1))[0]]
            return mom1 * (-4 + 8*n1)/n1**3 + mom2 * (2*(1 - 2*n1)**2)/n1**3 + mom3 * (1 - 2*n1)**2/(4.*n1**2) + mom4 * (-1 + 2*n1)/(4.*n1**2) + mom5 * (-1 + 2*n1)/(4.*n1**2) + 1. * 1/(4.*n1**2)
        elif popp1 == popp2: 
            if popq1 == popp1: # e.g. 1_1_1_2
                n1 = sample_sizes[int(popp1)-1]
                n2 = sample_sizes[int(popq2)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{1}'.format(popp1,popq2))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{0}_{1}'.format(popp1,popq2))[0]]
                mom3 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * (-2 + 4*n1)/n1**2 + mom2 * (1 - 1/(2.*n1)) + mom3 * 1/(2.*n1)
            elif popq2 == popp1: # e.g. 2_2_1_2
                n1 = sample_sizes[int(popp1)-1]
                n2 = sample_sizes[int(popq1)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{0}_{1}'.format(popp1,popq1))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{1}_{0}'.format(popp1,popq1))[0]]
                mom3 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * (-2 + 4*n1)/n1**2 + mom2 * (1 - 1/(2.*n1)) + mom3 * 1/(2.*n1)
            elif popq1 == popq2: # e.g. 1_1_2_2
                n1 = sample_sizes[int(popp1)-1]
                n2 = sample_sizes[int(popq1)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{1}_{1}'.format(popp1,popq1))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                mom3 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * ((-1 + 2*n1)*(-1 + 2*n2))/(4.*n1*n2) + mom2 * (-1 + 2*n1)/(4.*n1*n2) + mom3 * (-1 + 2*n2)/(4.*n1*n2) + 1. * 1/(4.*n1*n2)
            else: # e.g. 1_1_2_3
                n1 = sample_sizes[int(popp1)-1]
                mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{0}_{1}_{2}'.format(popp1,popq1,popq2))[0]]
                mom2 = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
                return mom1 * (1 - 1/(2.*n1)) + mom2 * 1/(2.*n1)
        else: # popp1 != popp2
            if popq1 == popq2:
                if popp1 == popq1: # e.g. 1_2_1_1
                    n1 = sample_sizes[int(popp1)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{0}_{0}'.format(popp1,popp2))[0]]
                    mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{0}'.format(popp1,popp2))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                    return mom1 * (1 - 1/(2.*n1)) + mom2 * (-2 + 4*n1)/n1**2 + mom3 * 1/(2.*n1)
                elif popp2 == popq1: # e.g. 1_2_2_2
                    n1 = sample_sizes[int(popp2)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{1}_{1}'.format(popp1,popp2))[0]]
                    mom2 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{0}'.format(popp2,popp1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                    return mom1 * (1 - 1/(2.*n1)) + mom2 * (-2 + 4*n1)/n1**2 + mom3 * 1/(2.*n1)
                else: # e.g. 1_2_3_3
                    n3 = sample_sizes[int(popq1)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{2}'.format(popp1,popp2,popq1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
                    return mom1 * (1 - 1/(2.*n3)) + mom2 * 1/(2.*n3)
            else: # popq1 != popq2
                if popp1 == popq1:
                    if popp2 == popq2: # e.g. 1_2_1_2
                        n1 = sample_sizes[int(popp1)-1]
                        n2 = sample_sizes[int(popp2)-1]
                        mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{0}_{1}'.format(popp1,popp2))[0]]
                        mom2 = stats[np.argwhere(np.array(stat_names) == 'DD_{0}_{1}'.format(popp1,popp2))[0]]
                        mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{1}'.format(popp1,popp2))[0]]
                        mom4 = stats[np.argwhere(np.array(stat_names) == 'Dz_{1}_{0}_{0}'.format(popp1,popp2))[0]]
                        return mom1 + mom2 * 4/(n1*n2) + mom3 * 2/n1 + mom4 * 2/n2
                    else: # e.g. 1_3_1_2 or 1_2_1_3
                        n1 = sample_sizes[int(popp1)-1]
                        mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{0}_{2}'.format(popp1,popp2,popq2))[0]]
                        mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp1,popp2,popq2))[0]]
                        return mom1 + mom2 * 2/n1
                elif popp1 == popq2: # e.g. 2_3_1_2
                    n2 = sample_sizes[int(popp1)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{0}'.format(popp1,popp2,popq1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp1,popp2,popq1))[0]]
                    return mom1 + mom2 * 2/n2
                elif popp2 == popq2: # e.g. 1_3_2_3
                    n3 = sample_sizes[int(popp2)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{1}'.format(popp1,popp2,popq1))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp2,popp1,popq1))[0]]
                    return mom1 + mom2 * 2/n3
                elif popp2 == popq1: # e.g. 1_2_2_3
                    n2 = sample_sizes[int(popp2)-1]
                    mom1 = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{1}_{2}'.format(popp1,popp2,popq2))[0]]
                    mom3 = stats[np.argwhere(np.array(stat_names) == 'Dz_{0}_{1}_{2}'.format(popp2,popp1,popq2))[0]]
                    return mom1 + mom2 * 2/n2
                else: # e.g. 1_2_3_4
                    mom = stats[np.argwhere(np.array(stat_names) == 'zz_{0}_{1}_{2}_{3}'.format(popp1,popp2,popq1,popq2))[0]]
                    return mom
    elif moment_type == 'zp':
        popp1 = name.split('_')[1]
        popp2 = name.split('_')[2]
        if popp1 == popp2:
            n1 = sample_sizes[int(popp1)-1]
            mom = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{0}'.format(popp1))[0]]
            return mom * (1 - 1/(2.*n1)) + 1/(2.*n1)
        else:
            mom = stats[np.argwhere(np.array(stat_names) == 'zp_{0}_{1}'.format(popp1,popp2))[0]]
            return mom
    elif moment_type == 'zq':
        popq1 = name.split('_')[1]
        popq2 = name.split('_')[2]
        if popq1 == popq2:
            n1 = sample_sizes[int(popq1)-1]
            mom = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{0}'.format(popq1))[0]]
            return mom * (1 - 1/(2.*n1)) + 1/(2.*n1)
        else:
            mom = stats[np.argwhere(np.array(stat_names) == 'zq_{0}_{1}'.format(popq1,popq2))[0]]
            return mom
    else:
        return -1e6
