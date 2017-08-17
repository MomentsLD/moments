import numpy as np
import scipy as sp
from scipy.sparse import linalg

import Spectrum_mod
import Numerics
import Jackknife as jk
import LinearSystem_2D as ls2

#------------------------------------------------------------------------------
# Functions for the computation of the Phi-moments for multidimensional models:
# we integrate the ode system on the Phi_n(i) to compute their evolution
# we write it (and solve it) as an approximated linear system:
#       Phi_n' = Bn(N) + (1/(4N)Dn + S1n + S2n)Phi_n
# where :
#       N is the total population size
#       Bn(N) is the mutation source term
#       1/(4N)Dn is the drift effect matrix
#       S1n is the selection matrix for h = 0.5
#       S2n is the effect of h != 0.5
#------------------------------------------------------------------------------

#-----------------------------------
# functions to compute the matrices-
#-----------------------------------
# Mutations
def _calcB(dims, u):
    """
    dims : List containing the pop sizes

    u : scalar mutation rate

    Returns a mutation matrix
    """ 
    B = np.zeros(dims)
    for k in range(len(dims)):
        ind = np.zeros(len(dims), dtype='int')
        ind[k] = int(1)
        tp = tuple(ind)
        B[tp] = dims[k] - 1
    return u * B

# Drift
def _calcD(dims):
    """
    dims : List containing the pop sizes

    Returns a list of drift matrices for each pair of pops
    """ 
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append([ls2.calcD1(np.array([dims[i], dims[j]])),
                        ls2.calcD2(np.array([dims[i], dims[j]]))])
    return res

def _buildD(vd, dims, N):
    """
    Builds the effective drift matrices by multiplying by the 1/4N coeff

    vd : List containing the drift matrices

    dims : List containing the pop sizes
    
    N : List containing the effective pop sizes for each pop 

    Returns a list of effective drift matrices for each pair of pops
    """ 
    if (len(dims) == 1): return [1.0 / 4 / N[0] * vd[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(1.0/(4*N[i])*vd[ctr][0] + 1.0/(4*N[j])*vd[ctr][1])
            ctr += 1
    return res

# Selection 1
def _calcS(dims, ljk):
    """
    dims : List containing the pop sizes

    ljk : List containing the 1 jump jackknife matrices for each pair of pop

    Returns a list of selection matrices for each pair of pops
    """ 
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append([ls2.calcS_1(np.array([dims[i], dims[j]]), ljk[i]),
                        ls2.calcS_2(np.array([dims[i], dims[j]]), ljk[j])])
    return res

def _buildS(vs, dims, s, h):
    """
    Builds the effective selection matrices by multiplying by the correct coeff

    vs : List containing the selection matrices

    dims : List containing the pop sizes
    
    s : List containing the selection coefficients 

    h : List containing the dominance coefficients 

    Returns a list of effective selection matrices for each pair of pops
    """ 
    if (len(dims) == 1): return [vs[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(s[i]*h[i]*vs[ctr][0] + s[j]*h[j]*vs[ctr][1])
            ctr += 1
    return res

# Selection 2
def _calcS2(dims, ljk):
    """
    dims : List containing the pop sizes

    ljk : List containing the 2 jumps jackknife matrices for each pair of pop

    Returns a list of selection matrices for each pair of pops
    """ 
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append([ls2.calcS2_1(np.array([dims[i], dims[j]]), ljk[i]),
                        ls2.calcS2_2(np.array([dims[i], dims[j]]), ljk[j])])
    return res

def _buildS2(vs, dims, s, h):
    """
    Builds the effective selection matrices (part due to dominance)
    by multiplying by the correct coeff

    vs : List containing the selection matrices

    dims : List containing the pop sizes
    
    s : List containing the selection coefficients 

    h : List containing the dominance coefficients 

    Returns a list of effective selection matrices for each pair of pops
    """ 
    if (len(dims) == 1): return [vs[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(s[i]*(1-2.0*h[i])*vs[ctr][0] + s[j]*(1-2.0*h[j])*vs[ctr][1])
            ctr += 1
    return res

# Migrations
def _calcM(dims, ljk):
    """
    dims : List containing the pop sizes

    ljk : List containing the 1 jump jackknife matrices for each pair of pop

    Returns a list of migration matrices for each pair of pops
    """ 
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append([ls2.calcM_1(np.array([dims[i], dims[j]]), ljk[j]),
                        ls2.calcM_2(np.array([dims[i], dims[j]]), ljk[i])])
    return res

def _buildM(vm, dims, m):
    """
    Builds the effective migration matrices by multiplying by the migration coeff

    vm : List containing the migration matrices

    dims : List containing the pop sizes
    
    m : matrix containing the migration coefficients

    Returns a list of effective migration matrices for each pair of pops
    """ 
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(m[i, j]*vm[ctr][0] + m[j, i]*vm[ctr][1])
            ctr += 1
    return res

#----------------------------------
# updates for the time integration-
#----------------------------------
# we solve a system like PX = QY
# step 1 functions correspond to the QY computation
# and step 2 to the resolution of PX = Y'

# 2D
#step 1
def _ud1_2pop_1(sfs, Q, dims):
    sfs = Q[0].dot(sfs.reshape(dims[0] * dims[1])).reshape(dims)
    return sfs

# step 2
def _ud2_2pop_1(sfs, slv, dims):
    sfs = (slv[0](sfs.reshape(dims[0] * dims[1]))).reshape(dims)
    return sfs

# for 3D, 4D and 5D cases, each couple of directions are coded separately to simplify the permutations...
#------------------------------
# 3D

# step 1
def _ud1_3pop_1(sfs, Q, dims):
    for i in range(int(dims[2])):
        sfs[:, :, i] = Q[0].dot(sfs[:, :, i].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
    return sfs

def _ud1_3pop_2(sfs, Q, dims):
    for i in range(int(dims[1])):
        sfs[:, i, :] = Q[1].dot(sfs[:, i, :].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
    return sfs

def _ud1_3pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        sfs[i, :, :] = Q[2].dot(sfs[i, :, :].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
    return sfs

# step 2
def _ud2_3pop_1(sfs, slv, dims):
    for i in range(int(dims[2])):
        sfs[:, :, i] = slv[0](sfs[:, :, i].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
    return sfs

def _ud2_3pop_2(sfs, slv, dims):
    for i in range(int(dims[1])):
        sfs[:, i, :] = slv[1](sfs[:, i, :].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
    return sfs

def _ud2_3pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        sfs[i, :, :] = slv[2](sfs[i, :, :].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
    return sfs


#------------------------------
# 4D

# step 1
def _ud1_4pop_1(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:, :, i, j] = Q[0].dot(sfs[:, :, i, j].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
    return sfs

def _ud1_4pop_2(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:, i, :, j] = Q[1].dot(sfs[:, i, :, j].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
    return sfs

def _ud1_4pop_3(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:, i, j, :] = Q[2].dot(sfs[:, i, j, :].reshape(dims[0] * dims[3])).reshape(dims[0], dims[3])
    return sfs

def _ud1_4pop_4(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i, :, :, j] = Q[3].dot(sfs[i, :, :, j].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
    return sfs

def _ud1_4pop_5(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i, :, j, :] = Q[4].dot(sfs[i, :, j, :].reshape(dims[1] * dims[3])).reshape(dims[1], dims[3])
    return sfs

def _ud1_4pop_6(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i, j, :, :] = Q[5].dot(sfs[i, j, :, :].reshape(dims[2] * dims[3])).reshape(dims[2], dims[3])
    return sfs

# step 2

def _ud2_4pop_1(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:, :, i, j] = slv[0](sfs[:, :, i, j].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
    return sfs

def _ud2_4pop_2(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:, i, :, j] = slv[1](sfs[:, i, :, j].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
    return sfs

def _ud2_4pop_3(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:, i, j, :] = slv[2](sfs[:, i, j, :].reshape(dims[0] * dims[3])).reshape(dims[0], dims[3])
    return sfs

def _ud2_4pop_4(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i, :, :, j] = slv[3](sfs[i, :, :, j].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
    return sfs

def _ud2_4pop_5(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i, :, j, :] = slv[4](sfs[i, :, j, :].reshape(dims[1] * dims[3])).reshape(dims[1], dims[3])
    return sfs

def _ud2_4pop_6(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i, j, :, :] = slv[5](sfs[i, j, :, :].reshape(dims[2] * dims[3])).reshape(dims[2], dims[3])
    return sfs

#------------------------------
# 5D

# step 1
def _ud1_5pop_1(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :, :] = Q[9].dot(sfs[i, j, k, :, :].reshape(dims[3] * dims[4])).reshape(dims[3], dims[4])
    return sfs

def _ud1_5pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k, :] = Q[8].dot(sfs[i, j, :, k, :].reshape(dims[2] * dims[4])).reshape(dims[2], dims[4])
    return sfs

def _ud1_5pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[4])):
                sfs[i, j, :, :, k] = Q[7].dot(sfs[i, j, :, :, k].reshape(dims[2] * dims[3])).reshape(dims[2], dims[3])
    return sfs

def _ud1_5pop_4(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k, :] = Q[6].dot(sfs[i, :, j, k, :].reshape(dims[1] * dims[4])).reshape(dims[1], dims[4])
    return sfs

def _ud1_5pop_5(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[i, :, j, :, k] = Q[5].dot(sfs[i, :, j, :, k].reshape(dims[1] * dims[3])).reshape(dims[1], dims[3])
    return sfs

def _ud1_5pop_6(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[i, :, :, j, k] = Q[4].dot(sfs[i, :, :, j, k].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
    return sfs

def _ud1_5pop_7(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k, :] = Q[3].dot(sfs[:, i, j, k, :].reshape(dims[0] * dims[4])).reshape(dims[0], dims[4])
    return sfs

def _ud1_5pop_8(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[:, i, j, :, k] = Q[2].dot(sfs[:, i, j, :, k].reshape(dims[0] * dims[3])).reshape(dims[0], dims[3])
    return sfs

def _ud1_5pop_9(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, i, :, j, k] = Q[1].dot(sfs[:, i, :, j, k].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
    return sfs

def _ud1_5pop_10(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, :, i, j, k] = Q[0].dot(sfs[:, :, i, j, k].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
    return sfs

# step 2
def _ud2_5pop_1(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :, :] = slv[9](sfs[i, j, k, :, :].reshape(dims[3] * dims[4])).reshape(dims[3], dims[4])
    return sfs

def _ud2_5pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k, :] = slv[8](sfs[i, j, :, k, :].reshape(dims[2] * dims[4])).reshape(dims[2], dims[4])
    return sfs

def _ud2_5pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[4])):
                sfs[i, j, :, :, k] = slv[7](sfs[i, j, :, :, k].reshape(dims[2] * dims[3])).reshape(dims[2], dims[3])
    return sfs

def _ud2_5pop_4(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k, :] = slv[6](sfs[i, :, j, k, :].reshape(dims[1] * dims[4])).reshape(dims[1], dims[4])
    return sfs

def _ud2_5pop_5(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[i, :, j, :, k] = slv[5](sfs[i, :, j, :, k].reshape(dims[1] * dims[3])).reshape(dims[1], dims[3])
    return sfs

def _ud2_5pop_6(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[i, :, :, j, k] = slv[4](sfs[i, :, :, j, k].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
    return sfs

def _ud2_5pop_7(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k, :] = slv[3](sfs[:, i, j, k, :].reshape(dims[0] * dims[4])).reshape(dims[0], dims[4])
    return sfs

def _ud2_5pop_8(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[:, i, j, :, k] = slv[2](sfs[:, i, j, :, k].reshape(dims[0] * dims[3])).reshape(dims[0], dims[3])
    return sfs

def _ud2_5pop_9(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, i, :, j, k] = slv[1](sfs[:, i, :, j, k].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
    return sfs

def _ud2_5pop_10(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, :, i, j, k] = slv[0](sfs[:, :, i, j, k].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
    return sfs

# update nD with permutations
def _update_step1(sfs, Q, dims, order):
    assert(len(sfs.shape) == len(dims))
    assert(len(Q) == len(dims) * (len(dims)-1) / 2)
    for i in order:
        sfs = eval('_ud1_' + str(len(dims)) + 'pop_' + str(i + 1) + '(sfs, Q, dims)')
    return sfs

def _update_step2(sfs, slv, dims, order):
    assert(len(sfs.shape) == len(dims))
    assert(len(slv) == len(dims) * (len(dims)-1) / 2)
    for i in order:
        sfs = eval('_ud2_' + str(len(dims)) + 'pop_' + str(i + 1) + '(sfs, slv, dims)')
    return sfs

def _permute(tab):
    res = tab[1:]
    res.append(tab[0])
    return res

# timestep computation 
def compute_dt(sample_sizes, N, s, h, m, dt_default, factor=10.0):
    sample_sizes = np.array(sample_sizes, dtype=np.float64)
    N = np.amin(np.array(N)/sample_sizes)
    sel1 = np.amax(abs(np.array(s) * np.array(h) * sample_sizes))
    sel2 = np.amax(abs(np.array(s) * (1-2.0*np.array(h)) * sample_sizes))
    mig = np.amax(m) * np.amax(sample_sizes)
    eps = 10e-16 # to avoid division by zero
    return min(dt_default, factor * min(2*N, 1.0 / (mig+eps),
               1.0 / (sel1+eps), 1.0 / (sel2+eps)))

def compute_dt_bis(N, T, drift, selmig, dims):
    if callable(N):
        Nmin = N(0)#np.amin(N(0), N(T))
    else:
        Nmin = N
    nbp = int(len(dims) * (len(dims)-1) / 2)
    D = _buildD(drift, dims, Nmin)
    Mat =  [(D[i] + selmig[i]).todense() for i in range(nbp)]
    ev = [np.linalg.eigvals(Mat[i]) for i in range(nbp)]
    return 0

def _compute_dt_1pop(N, m, s, h, timescale_factor=0.15):
    maxVM = max(0.25/N, max(m),\
                abs(s) * 2*max(np.abs(h + (1-2*h)*0.5) * 0.5*(1-0.5),
                                   np.abs(h + (1-2*h)*0.25) * 0.25*(1-0.25)))
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = np.inf
    if dt == 0:
        raise ValueError('Timestep is zero. Values passed in are N=%f, m=%s,'
                         's=%f, h=%f.' % (N, str(m), s, h))
    return dt

def compute_dt(N, m=None, s=None, h=None, timescale_factor=0.1):
#def compute_dt(N, m=None, s=None, h=None, timescale_factor=0.05):
    if m is None:
        m = np.zeros([len(N), len(N)])
    if s is None:
        s = np.zeros(len(N))
    if h is None:
        h = 0.5*np.ones(len(N))
    timesteps = [_compute_dt_1pop(N[i], m[i, :], s[i], h[i], timescale_factor) for i in range(len(N))]
    return min(timesteps)




def integrate_nD(sfs0, Npop, tf, dt_fac=0.1, gamma=None, h=None, m=None, theta=1.0, adapt_dt=False):
    """
#--------------------
# Integration in time
#--------------------
# N : total population size (vector N = (N1,...,Np))
# tf : final simulation time (/2N1 generations)
# gamma : selection coefficients (vector gamma = (gamma1,...,gammap))
# theta : mutation rate
# h : allele dominance (vector h = (h1,...,hp))
# m : migration rates matrix (2D array, m[i,j] is the migration rate from pop j to pop i, normalized by 1/4N1)

# for a "lambda" definition of N - with backward Euler integration scheme
# where t is the relative time in generations such as t = 0 initially
# Npop is a lambda function of the time t returning the vector N = (N1,...,Np) or directly the vector if N does not evolve in time
    """
    
    
    
    # neutral case if the parameters are not provided
    if gamma is None: gamma = np.zeros(len(n))
    if h is None: h = 0.5 * np.ones(len(n))
    if m is None: m = np.zeros([len(n), len(n)])
    
    sfs0 = np.array(sfs0)
    n = np.array(sfs0.shape)-1
    # parameters of the equation
    if callable(Npop): 
        N = np.array(Npop(0))
    else: 
        N = np.array(Npop)
    
    Nold = np.ones(len(N))
    Neff = N
    mm = np.array(m) / 2.0
    s = np.array(gamma)
    h = np.array(h)
    Tmax = tf * 2.0
    dt = Tmax * dt_fac
    u = theta / 4.0
    # dimensions of the sfs
    dims = np.array(n + np.ones(len(n)), dtype=int)
    d = int(np.prod(dims))
    # number of "directions" for the splitting
    nbp = int(len(n) * (len(n)-1) / 2)
    if len(n) == 1: nbp = 1
    # we compute the matrices we will need
    ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(len(dims))]
    ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(len(dims))]
    
    # drift
    vd = _calcD(dims)
    D = _buildD(vd, dims, N)
    
    # selection part 1
    vs = _calcS(dims, ljk)
    S1 = _buildS(vs, dims, s, h)
    
    # selection part 2
    vs2 = _calcS2(dims, ljk2)
    S2 = _buildS2(vs2, dims, s, h)
    
    # migration
    vm = _calcM(dims, ljk)
    Mi = _buildM(vm, dims, mm)
    
    # mutations
    B = _calcB(dims, u)
    
    # indexes for the permutation trick
    order = list(range(nbp))

    # time step splitting
    split_dt = 1.0
    if len(n) > 2: split_dt = 2.0*len(n)

    # indicator of negative entries
    neg = False

    # time loop:
    t = 0.0
    sfs = sfs0
    
    while t < Tmax:
        dt_old = dt
        sfs_old = sfs
        if neg == False:
            dt = min(compute_dt(N, mm, s, h), Tmax * dt_fac)
        if t+dt > Tmax:
            dt = Tmax-t
        # we update the value of N if a function was provided as argument
        if callable(Npop):
            N = np.array(Npop((t+dt) / 2.0))
            Neff = Numerics.compute_N_effective(Npop, 0.5*t, 0.5*(t+dt))

        # we recompute the matrix only if N has changed...
        if t == 0.0 or (Nold != N).any() or dt != dt_old or neg == True:
            D = _buildD(vd, dims, Neff)
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype='float', format='csc') 
                   - dt/2.0/split_dt*(1.0/(max(len(n), 2)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(nbp)]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype='float', format='csc')
                 + dt/2.0/split_dt*(1.0/(max(len(n), 2)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in range(nbp)]
            
        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs + dt*B)
        elif len(n) > 1:
            for i in range(int(split_dt)):
                sfs = _update_step1(sfs, Q, dims, order)
                sfs += dt / split_dt * B
                sfs = _update_step2(sfs, slv, dims, order)
                order = _permute(order)
        
        if (sfs<0).any() and adapt_dt:
            neg = True
            if dt > min(compute_dt(N, mm, s, h), Tmax * dt_fac) / 8.0:
                dt*=0.5
            sfs = sfs_old

        else:
            neg = False
            Nold = N
            t += dt
        
        Nold = N
        t += dt

    return Spectrum_mod.Spectrum(sfs)
