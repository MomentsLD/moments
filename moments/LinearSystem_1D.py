import numpy as np
import scipy as sp
from scipy.sparse import linalg

import Jackknife as jk
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
#-------------------------------------------------------------------------------

# We compute the  matrices for drift
# this function returns a list of matrices corresponding to each population
# dims -> array containing the dimensions of the problem dims[j] = nj+1
def calcD(dims):
    # number of freedom degrees
    d = int(dims[0])
    # we consider separately the contributions of each dimension
    data = []
    row = []
    col = []

    # loop over the fs elements:
    for i in range(d):
        if (i>1):
            data.append((i-1)*(d-i))
            row.append(i)
            col.append(i-1)
        if (i<d-2):
            data.append((i+1)*(d-i-2))
            col.append(i+1)
            row.append(i)
        if (i>0) and (i<d-1):
            data.append(-2*i*(d-i-1))
            row.append(i)
            col.append(i)

    return sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()

# Selection
def calcS(dims, ljk):
    # number of degrees of freedom
    d = int(dims[0])

    data = []
    row = []
    col = []
    for i in range(d):
        i_bis = jk.index_bis(i,d-1)
        i_ter = jk.index_bis(i+1,d-1)

        g1 = i*(d-i)/np.float64(d)
        g2 = -(i+1)*(d-1-i)/np.float64(d)

        if (i<d-1):
            data += [g1*ljk[i-1,i_bis-1], g1*ljk[i-1,i_bis-2],
                    g1*ljk[i-1,i_bis], g2*ljk[i,i_ter-1],
                    g2*ljk[i,i_ter-2], g2*ljk[i,i_ter]]
            row += [i, i, i, i, i, i]
            col += [i_bis, i_bis-1, i_bis+1,
                    i_ter, i_ter-1, i_ter+1]
            
        if i==d-1: # g2=0
            data += [g1*ljk[i-1,i_bis-1], g1*ljk[i-1,i_bis-2], g1*ljk[i-1,i_bis]]
            row += [i, i, i]
            col += [i_bis, i_bis-1, i_bis+1]

    return sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()


# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
def calcS2(dims, ljk):
    # number of degrees of freedom
    d = int(dims[0])

    data = []
    row = []
    col = []
    for i in range(d):
        i_ter = jk.index_bis(i+1,d-1)
        i_qua = jk.index_bis(i+2,d-1)

        g1 = (i+1)/np.float64(d)/(d+1.0)*i*(d-i)
        g2 = -(i+1)/np.float64(d)/(d+1.0)*(i+2)*(d-1-i)

        if i<d-1:
            data += [g1*ljk[i,i_ter-1], g1*ljk[i,i_ter-2],
                    g1*ljk[i,i_ter], g2*ljk[i+1,i_qua-1],
                    g2*ljk[i+1,i_qua-2], g2*ljk[i+1,i_qua]]
            row += [i, i, i, i, i, i]
            col += [i_ter, i_ter-1, i_ter+1,
                    i_qua, i_qua-1, i_qua+1]
            
        if i==d-1: # g2=0
            data += [g1*ljk[i,i_ter-1], g1*ljk[i,i_ter-2], g1*ljk[i,i_ter]]
            row += [i, i, i]
            col += [i_ter, i_ter-1, i_ter+1]

    return sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()

#----------------------------------
# Steady state (for initialization)
#----------------------------------
def steady_state_1D(n, N=1.0, gamma=0.0, h=0.5, theta=1.0):

    # dimensions of the sfs
    d = n+1
    
    # matrix for mutations
    u = np.float64(theta/4.0)
    s = np.float64(gamma)

    B = np.zeros(d)
    B[1] = n*u

    # matrix for drift
    D = 1/4.0/N*calcD([d])
    # jackknife matrices
    ljk = jk.calcJK13(int(d-1))
    ljk2 = jk.calcJK23(int(d-1))
    # matrix for selection
    S = calcS([d], ljk, s, h)
    #S = s*h*calcS(dims, ljk, s, h)
    
    S2 = calcS2([d], ljk2, s, h)
    #S2 = s*(1-2.0*h)*calcS(dims, ljk, s, h)
    # matrix for migration
    Mat = D+S+S2

    sfs = sp.sparse.linalg.spsolve(Mat[1:d-1,1:d-1],-B[1:d-1])
    sfs = np.insert(sfs, 0, 0.0)
    sfs = np.insert(sfs, d-1, 0.0)

    return sfs
