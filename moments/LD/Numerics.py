import numpy as np

from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as spinv

import Matrices

import networkx as nx
import pickle
import itertools

### one pop numerics

# for a given order D^n
names = {}
lengths = {}
def moment_names(n):
    n = int(n)
    try:
        moments = names[n]
    except KeyError:
        moments = []
        if n%2: # odd
            order = 1
            last_moments = ['D^1','z']
            moments = last_moments + moments
        else: # even
            order = 0
            last_moments = ['1']
            moments = last_moments + moments
        while order < n:
            order += 2
            last_sigma = []
            for mom in last_moments:
                if '_s' not in mom:
                    new_mom = mom+'_s1'
                else:
                    s_order = int(mom.split('_s')[1])
                    new_mom = mom.split('_s')[0]+'_s{0}'.format(s_order+1)
                last_sigma.append(new_mom)
            new_moments = []
            for ii in range(order+1):
                pi_order = ii/2
                z_order = ii%2
                if pi_order > 0:
                    if z_order > 0:
                        appendage = '_pi^{0}_z'.format(pi_order)
                    else:
                        appendage = '_pi^{0}'.format(pi_order)
                else:
                    if z_order > 0:
                        appendage = '_z'
                    else:
                        appendage = ''
                if ii < order:
                    new_mom = 'D^{0}'.format(order-ii) + appendage
                else:
                    new_mom = appendage[1:]
                new_moments.append(new_mom)
            moments = new_moments + last_sigma + moments
            last_moments = new_moments + last_sigma
        names[n] = moments
        lengths[len(moments)] = n
    return moments

def drift(n):
    order = int(n)
    row = []
    col = []
    data = []
    corner = 0
    while order >= 0:
        row_current = Matrices.drift_dict[order]['row']
        new_row = [x+corner for x in row_current]
        col_current = Matrices.drift_dict[order]['col']
        new_col = [x+corner for x in col_current]
        data_current = Matrices.drift_dict[order]['data']
        size = Matrices.drift_dict[order]['size']
        row.extend(new_row)
        col.extend(new_col)
        data.extend(data_current)
        corner += size[0]
        order -= 2
    return csc_matrix((data,(row,col)),shape=(corner,corner))

def mutation(n, ism):
    order = int(n)
    row = []
    col = []
    data = []
    if ism == False:
        corner = 0
        while order >= 0:
            row_current = Matrices.mut_dict[order]['row']
            new_row = [x+corner for x in row_current]
            col_current = Matrices.mut_dict[order]['col']
            new_col = [x+corner for x in col_current]
            data_current = Matrices.mut_dict[order]['data']
            size = Matrices.mut_dict[order]['size']
            row.extend(new_row)
            col.extend(new_col)
            data.extend(data_current)
            corner += size[0]
            order -= 2
        return csc_matrix((data,(row,col)),shape=(corner,corner))
    elif ism == True:
        # ISM model only built for even orders
        names = moment_names(n)
        size = len(names)
        M = np.zeros((size,size))
        # [pi s_{i}]_{t+1} = [pi s_{i}]_{t} + theta/2 [s_{i+1}]_{t}
        M[names.index('1_s1'), names.index('1')] = 1.0
        M[names.index('pi^1'), names.index('1_s1')] = 1./2
        for ii in range(1,n/2):
            M[names.index('pi^1_s{0}'.format(ii)), names.index('1_s{0}'.format(ii+1))] = 1./2
        
        return csc_matrix(M)

def recombination(n):
    row = []
    data = []
    moms = moment_names(n)
    for ii,moment in zip(range(len(moms)),moms):
        if 'D' in moment:
            D_order = int(moment.split('_')[0].split('^')[1])
            row.append(ii)
            data.append(-D_order/2.)
        else:
            continue
    size = (len(moms),len(moms))
    return csc_matrix((data,(row,row)),shape=size)

def integrate(y, T, rho=0.0, nu=1.0, theta=0.0008, order=None, dt=0.001, ism=False):
    if order is None:
        try:
            order = lengths[len(y)]
        except KeyError:
            raise KeyError("specify order or get moment names")
    
    moms = moment_names(order)
    if len(y) != len(moms):
        raise ValueError("there is a vector size mismatch")

    
#    D = drift(order)
#    M = mutation(order, ism)
#    R = recombination(order)
        
    D = drift(order).toarray()
    M = mutation(order, ism).toarray()
    R = recombination(order).toarray()
    EYE = np.eye(D.shape[0])
    
    N = 1.0
    
    elapsed_t = 0
    while elapsed_t < T:
        # ensure that final integration time does not exceed T
        if elapsed_t + dt > T:
            dt = T-elapsed_t
        
        # if nu is a function, set N to nu(t+dt/2)
        if callable(nu):
            N = nu(elapsed_t + dt/2.)
        else:
            N = nu
        
        if elapsed_t == 0 or dt != dt_old or N != N_old:
            A = D/N + M*theta + R*rho
#            Afd = identity(A.shape[0]) + dt/2.*A
            Afd = EYE + dt/2.*A
#            Abd = factorized(identity(A.shape[0]) - dt/2.*A)
            Abd = np.linalg.inv(EYE - dt/2.*A)
        
#        y = Abd(Afd.dot(y))
        y = Abd.dot(Afd.dot(y))
        elapsed_t += dt
        dt_old = dt
        N_old = N
    
    return y

def equilibrium(order=2, rho=0.0, theta=0.0008, ism=False):
    D = drift(order)
    R = recombination(order)
    M = mutation(order, ism)
    A = D + M*theta + R*rho
    B = A[:-1,-1]
    A = A[:-1,:-1]
    return np.concatenate(( np.ndarray.flatten(np.array(-factorized(A)(B.todense()))), np.array([1]) ))


### multi pop numerics
