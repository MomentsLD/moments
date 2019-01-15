import numpy as np
import Util

from scipy.sparse import csc_matrix

## matrices for new LDstats2 models (h and compressed ys)


### drift

def drift_h(num_pops, nus):
    if num_pops != len(nus): raise ValueError("number of pops must match length of nus.")
    D = np.zeros( ( int(num_pops*(num_pops+1)/2), int(num_pops*(num_pops+1)/2) ) )
    c = 0
    for ii in range(num_pops):
        D[c,c] = -1./nus[ii]
        c += (num_pops-ii)
    return D

def drift_ld(num_pops, nus):
    names = Util.ld_names(num_pops)
    row = []
    col = []
    data = []
    for ii, name in enumerate(names):
        mom = name.split('_')[0]
        pops = name.split('_')[1:]
        if mom == 'DD':
            pop1, pop2 = [int(p) for p in pops]
            if pop1 == pop2:
                new_rows = [ii, ii, ii]
                new_cols = [names.index('DD_{0}_{0}'.format(pop1)), 
                            names.index('Dz_{0}_{0}_{0}'.format(pop1)),
                            names.index('pi2_{0}_{0}_{0}_{0}'.format(pop1))]
                new_data = [-3./nus[pop1-1], 1./nus[pop1-1], 1./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            else:
                row.append( ii )
                col.append( names.index('DD_{0}_{1}'.format(pop1, pop2)) )
                data.append( - 1./nus[pop1-1] - 1./nus[pop2-1] )
        elif mom == 'Dz':
            pop1, pop2, pop3 = [int(p) for p in pops]
            if pop1 == pop2 == pop3:
                new_rows = [ii, ii]
                new_cols = [names.index('DD_{0}_{0}'.format(pop1)),
                            names.index('Dz_{0}_{0}_{0}'.format(pop1))]
                new_data = [4./nus[pop1-1], -5./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2:
                row.append( ii )
                col.append( names.index('Dz_{0}_{1}_{2}'.format(pop1, pop2, pop3)) )
                data.append( -3./nus[pop1-1] )
            elif pop1 == pop3:
                row.append( ii )
                col.append( names.index('Dz_{0}_{1}_{2}'.format(pop1, pop2, pop3)) )
                data.append( -3./nus[pop1-1] )
            elif pop2 == pop3:
                new_rows = [ii, ii]
                new_cols = [names.index(Util.map_moment('DD_{0}_{1}'.format(pop1,pop2))),
                            names.index('Dz_{0}_{1}_{1}'.format(pop1,pop2))]
                new_data = [4./nus[pop2-1], -1./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            else: # all different
                row.append(ii)
                col.append(names.index('Dz_{0}_{1}_{2}'.format(pop1,pop2,pop3)))
                data.append(-1./nus[pop1-1])
        elif mom == 'pi2':
            pop1, pop2, pop3, pop4 = [int(p) for p in pops]
            if pop1 == pop2 == pop3 == pop4:
                new_rows = [ii, ii]
                new_cols = [names.index('Dz_{0}_{0}_{0}'.format(pop1)),
                            names.index('pi2_{0}_{0}_{0}_{0}'.format(pop1))]
                new_data = [1./nus[pop1-1], -2./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2 == pop3:
                new_rows = [ii, ii]
                new_cols = [names.index('Dz_{0}_{0}_{1}'.format(pop1,pop4)),
                            names.index('pi2_{0}_{0}_{0}_{1}'.format(pop1,pop4))]
                new_data = [1./2/nus[pop1-1], -1./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2 == pop4:
                new_rows = [ii, ii]
                new_cols = [names.index('Dz_{0}_{1}_{0}'.format(pop1,pop3)),
                            names.index('pi2_{0}_{0}_{1}_{0}'.format(pop1,pop3))]
                new_data = [1./2/nus[pop1-1], -1./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2 and pop3 == pop4:
                row.append( ii )
                col.append( names.index('pi2_{0}_{0}_{1}_{1}'.format(pop1,pop3)) )
                data.append( - 1./nus[pop1-1] - 1./nus[pop3-1] )
            elif pop1 == pop2:
                row.append( ii )
                col.append( names.index('pi2_{0}_{0}_{1}_{2}'.format(pop1,pop3,pop4)) )
                data.append( -1./nus[pop1-1] )
            elif pop1 == pop3 == pop4:
                new_rows = [ii, ii]
                new_cols = [names.index(Util.map_moment('Dz_{0}_{1}_{0}'.format(pop1,pop2))),
                            names.index('pi2_{0}_{1}_{0}_{0}'.format(pop1,pop2))]
                new_data = [1./2/nus[pop1-1], -1./nus[pop1-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop2 == pop3 == pop4:
                new_rows = [ii, ii]
                new_cols = [names.index(Util.map_moment('Dz_{1}_{0}_{1}'.format(pop1,pop2))),
                            names.index('pi2_{0}_{1}_{1}_{1}'.format(pop1,pop2))]
                new_data = [1./2/nus[pop2-1], -1./nus[pop2-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop3 and pop2 == pop4:
                new_rows = [ii, ii]
                new_cols = [names.index('Dz_{0}_{1}_{1}'.format(pop1,pop2)),
                            names.index('Dz_{1}_{0}_{0}'.format(pop1,pop2))]
                new_data = [-1./4/nus[pop1-1], 1./4/nus[pop2-1]]
                for r,c,d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop3:
                row.append( ii )
                col.append( names.index(Util.map_moment('Dz_{0}_{1}_{2}'.format(pop1,pop2,pop4))) )
                data.append( 1./4/nus[pop1-1] )
            elif pop1 == pop4:
                row.append( ii )
                col.append( names.index(Util.map_moment('Dz_{0}_{1}_{2}'.format(pop1,pop2,pop3))) )
                data.append( 1./4/nus[pop1-1] )
            elif pop2 == pop3:
                row.append( ii )
                col.append( names.index(Util.map_moment('Dz_{1}_{0}_{2}'.format(pop1,pop2,pop4))) )
                data.append( 1./4/nus[pop2-1] )
            elif pop2 == pop4:
                row.append( ii )
                col.append( names.index(Util.map_moment('Dz_{1}_{0}_{2}'.format(pop1,pop2,pop3))) )
                data.append( 1./4/nus[pop2-1] )
            elif pop3 == pop4:
                row.append( ii )
                col.append( names.index('pi2_{0}_{1}_{2}_{2}'.format(pop1,pop2,pop3)) )
                data.append( -1./nus[pop3-1] )
            else:
                if len(set([pop1,pop2,pop3,pop4])) < 4:
                    print("oh no")
                    print(pop1,pop2,pop3,pop4)
            
    return csc_matrix((data,(row,col)),shape=(len(names), len(names)))

### mutation
def mutation_h(num_pops, theta, frozen=None):
    if frozen is None:
        return theta*np.ones(int(num_pops*(num_pops+1)/2))
    else:
        U = np.zeros( int(num_pops*(num_pops+1)/2) )
        c = 0
        for ii in range(num_pops):
            for jj in range(ii,num_pops):
                if frozen[ii] is not True:
                    U[c] += theta/2.
                if frozen[jj] is not True:
                    U[c] += theta/2.
                c += 1
        return U

def mutation_ld(num_pops, theta, frozen=None):
    names_ld, names_h = Util.moment_names(num_pops)
    row = []
    col = []
    data = []
    for ii, mom in enumerate(names_ld):
        name = mom.split('_')[0]
        if name == 'pi2': 
            hmomp = 'H_' + mom.split('_')[1] +'_' + mom.split('_')[2]
            hmomq = 'H_' + mom.split('_')[3] +'_' + mom.split('_')[4]
            if hmomp == hmomq:
                row.append(ii)
                col.append(names_h.index(hmomp))
                data.append(theta/2.)
            else:
                row.append(ii)
                row.append(ii)
                col.append(names_h.index(hmomp))
                col.append(names_h.index(hmomq))
                data.append(theta/4.)
                data.append(theta/4.)
    return csc_matrix((data,(row,col)),shape=(len(names_ld), len(names_h)))
    
### recombination

def recombination(num_pops, r):
    names = Util.ld_names(num_pops)
    row = list(range(int(num_pops*(num_pops+1)/2 + num_pops**2*(num_pops+1)/2)))
    col = list(range(int(num_pops*(num_pops+1)/2 + num_pops**2*(num_pops+1)/2)))
    data = [-1.*r]*int(num_pops*(num_pops+1)/2) + [-r/2.]*int(num_pops**2*(num_pops+1)/2)
    return csc_matrix((data,(row,col)),shape=(len(names), len(names)))

### migration


"""
def migration_h(mig_mat):
    num_pops = len(mig_mat)
    M = np.zeros( ( int(num_pops*(num_pops+1)/2), int(num_pops*(num_pops+1)/2) ) )
    Hs = het_names(num_pops)
    for ii,H in enumerate(Hs):
        pop1,pop2 = [int(f) for f in H.split('_')[1:]]
        if pop1 == pop2:
            for jj in range(1,num_pops+1):
                if jj == pop1:
                    continue
                else:
                    M[ii,ii] -= 2*mig_mat[jj-1][pop1-1]
                    if pop1 < jj:
                        M[ii,Hs.index('H_{0}_{1}'.format(pop1,jj))] += 2*mig_mat[jj-1][pop1-1]
                    else:
                        M[ii,Hs.index('H_{0}_{1}'.format(jj,pop1))] += 2*mig_mat[jj-1][pop1-1]
        else:
            for jj in range(1,num_pops+1):
                if jj == pop1:
                    continue
                else:
                    M[ii,ii] -= mig_mat[jj-1][pop1-1]
                    if pop2 <= jj:
                        M[ii,Hs.index('H_{0}_{1}'.format(pop2,jj))] += mig_mat[jj-1][pop1-1]
                    else:
                        M[ii,Hs.index('H_{0}_{1}'.format(jj,pop2))] += mig_mat[jj-1][pop1-1]
            for jj in range(1,num_pops+1):
                if jj == pop2:
                    continue
                else:
                    M[ii,ii] -= mig_mat[jj-1][pop2-1]
                    if pop1 <= jj:
                        M[ii,Hs.index('H_{0}_{1}'.format(pop1,jj))] += mig_mat[jj-1][pop2-1]
                    else:
                        M[ii,Hs.index('H_{0}_{1}'.format(jj,pop1))] += mig_mat[jj-1][pop2-1]
    
    return M

def migration_ld(num_pops, m):
    pass
"""
