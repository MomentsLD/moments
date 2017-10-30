import numpy as np
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import factorized

index_cache = {}
def index_n(n,i,j,k):
    """
    For a spectrum of sample size n, take in an (n+1)^3 sized object, convert to correctly sized array Phi.
    Here, we'll try to operate on the whole spectrum, since recombination cannot be split.
    We need a dictionary that maps (i,j,k) to the correct index in the array.
    """
    try:
        return index_cache[n][(i,j,k)]
    except KeyError:
        indexes = {}
        indexes.setdefault(n,{})
        ll = 0
        for ii in range(n+1):
            for jj in range(n+1-ii):
                for kk in range(n+1-ii-jj):
                    indexes[n][(ii,jj,kk)] = ll
                    ll += 1
        index_cache[n] = indexes[n]
        return index_cache[n][(i,j,k)]

def array_to_Phi(F):
    n = len(F)-1
    Phi = np.zeros((n+1)*(n+2)*(n+3)/6)
    for ii in range(n+1):
        for jj in range(n+1-ii):
            for kk in range(n+1-ii-jj):
                Phi[index_n(n,ii,jj,kk)] = F[ii,jj,kk]
    return Phi

def Phi_to_array(Phi,n):
    F = np.zeros((n+1,n+1,n+1))
    for ii in range(n+1):
        for jj in range(n+1-ii):
            for kk in range(n+1-ii-jj):
                F[ii,jj,kk] = Phi[index_n(n,ii,jj,kk)]
    return F

# so now we need slice drift transition matrices for each slice (3*(n+1) of them)
# we track the background biallelic frequency spectra along the [0,:,0] and [0,0,:] axes,
# so we don't want density flowing back onto those points

def choose(n,i):
    return np.exp(gammaln(n+1)- gammaln(n-i+1) - gammaln(i+1))

def drift(n):
    Dsize = (n+1)*(n+2)*(n+3)/6
    row = []
    col = []
    data = []
    for ii in range(n+1):
        for jj in range(n+1-ii):
            for kk in range(n+1-ii-jj):
                # skip if A/a or B/b fixed
                fA = ii + jj
                fa = n - (ii + jj)
                fB = ii + kk
                fb = n - (ii + kk)
                if fA == n or fB == n: continue
                # single locus drift along B/b and A/a variable, with other fixed ancestral
                # fixed for b
                elif fA == 0: # 0 <= kk <= n
                    if kk == 0 or kk == n: continue
                    this_ind = index_n(n,ii,jj,kk)
                    row.append(this_ind)
                    col.append(index_n(n,ii,jj,kk-1))
                    data.append(2*choose(n,2)*(1.*(kk-1)*(n-kk+1)/n/(n-1)))
                    row.append(this_ind)
                    col.append(index_n(n,ii,jj,kk))
                    data.append(2*choose(n,2)*(-2.*kk*(n-kk)/n/(n-1)))
                    row.append(this_ind)
                    col.append(index_n(n,ii,jj,kk+1))
                    data.append(2*choose(n,2)*(1.*(kk+1)*(n-kk-1)/n/(n-1)))
                elif fB == 0:
                    if jj == 0 or jj == n: continue
                    this_ind = index_n(n,ii,jj,kk)
                    row.append(this_ind)
                    col.append(index_n(n,ii,jj-1,kk))
                    data.append(2*choose(n,2)*(1.*(jj-1)*(n-jj+1)/n/(n-1)))
                    row.append(this_ind)
                    col.append(index_n(n,ii,jj,kk))
                    data.append(2*choose(n,2)*(-2.*jj*(n-jj)/n/(n-1)))
                    row.append(this_ind)
                    col.append(index_n(n,ii,jj+1,kk))
                    data.append(2*choose(n,2)*(1.*(jj+1)*(n-jj-1)/n/(n-1)))
                else:
                    this_ind = index_n(n,ii,jj,kk)
                    # incoming density
                    if ii > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii-1,jj,kk))
                        data.append(2*choose(n,2)*(1.*(ii-1)*(n-ii-jj-kk+1)/n/(n-1)))
                    
                    if n-ii-jj-kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii+1,jj,kk))
                        data.append(2*choose(n,2)*(1.*(ii+1)*(n-ii-jj-kk-1)/n/(n-1)))
                    
                    if ii > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii-1,jj,kk+1))
                        data.append(2*choose(n,2)*(1.*(ii-1)*(kk+1)/n/(n-1)))
                    
                    if kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii+1,jj,kk-1))
                        data.append(2*choose(n,2)*(1.*(ii+1)*(kk-1)/n/(n-1)))
                    
                    if ii > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii-1,jj+1,kk))
                        data.append(2*choose(n,2)*(1.*(ii-1)*(jj+1)/n/(n-1)))
                    
                    if jj > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii+1,jj-1,kk))
                        data.append(2*choose(n,2)*(1.*(ii+1)*(jj-1)/n/(n-1)))
                    
                    if jj > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii,jj-1,kk))
                        data.append(2*choose(n,2)*(1.*(jj-1)*(n-ii-jj-kk+1)/n/(n-1)))
                    
                    if n-ii-jj-kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii,jj+1,kk))
                        data.append(2*choose(n,2)*(1.*(jj+1)*(n-ii-jj-kk-1)/n/(n-1)))
                    
                    if jj > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii,jj-1,kk+1))
                        data.append(2*choose(n,2)*(1.*(jj-1)*(kk+1)/n/(n-1)))
                    
                    if kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii,jj+1,kk-1))
                        data.append(2*choose(n,2)*(1.*(jj+1)*(kk-1)/n/(n-1)))
                    
                    if kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii,jj,kk-1))
                        data.append(2*choose(n,2)*(1.*(kk-1)*(n-ii-jj-kk+1)/n/(n-1)))
                    
                    if n-ii-jj-kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n,ii,jj,kk+1))
                        data.append(2*choose(n,2)*(1.*(kk+1)*(n-ii-jj-kk-1)/n/(n-1)))
                    
                    #outgoing density
                    row.append(this_ind)
                    col.append(this_ind)
                    data.append(-2*choose(n,2) * 2.*(ii*(n-ii-jj-kk) + ii*kk + ii*jj + jj*(n-ii-jj-kk) + jj*kk + kk*(n-ii-jj-kk))/n/(n-1))
                
    return csc_matrix((data,(row,col)),shape=(Dsize,Dsize))

def mutations(n, theta=1.0):
    """
    Mutations can occur on a background with ???? huh?
    """
    Msize = (n+1)*(n+2)*(n+3)/6
    
    M_1to2 = np.zeros((Msize,Msize))
    # A/a -> AB and aB
    for j in range(0,n-1):
        M_1to2[index_n(n,1,j,0),index_n(n,0,j+1,0)] += (j+1)*theta/2.
    for j in range(1,n-1):
        M_1to2[index_n(n,0,j,1),index_n(n,0,j,0)] += (n-j)*theta/2.
    # B/b -> AB and Ab
    for k in range(0,n-1):
        M_1to2[index_n(n,1,0,k),index_n(n,0,0,k+1)] += (k+1)*theta/2.
    for k in range(1,n-1):
        M_1to2[index_n(n,0,1,k),index_n(n,0,0,k)] += (n-k)*theta/2.
    
    M_0to1 = np.zeros(Msize)
    M_0to1[index_n(n,0,0,1)] = M_0to1[index_n(n,0,1,0)] = n*theta/2.
    
    return M_0to1, csc_matrix(M_1to2)

def recombination(n, rho): #### XXXX !!!!! starting on surfaces should push density into interior with rho>0, but right now, the off-axis surface doesn't have this behavior (8/15) - is this still the case? (10/20)
    """
    rho = 4*Ne*r
    where r is the recombination probability
    """
    Rsize0 = (n+1)*(n+2)*(n+3)/6
    Rsize1 = (n+2)*(n+3)*(n+4)/6
    row = []
    col = []
    data = [] 
    
    for i in range(n+1):
        for j in range(n+1-i):
            for k in range(n+1-i-j):
                fA = i+j
                fa = n-i-j
                fB = i+k
                fb = n-i-k
                if fA == 0 or fa == 0 or fB == 0 or fb == 0:
                    continue
                
                # incoming
                if j > 0:
                    row.append( index_n(n,i,j,k) )
                    col.append( index_n(n+1,i+1,j-1,k) )
                    data.append( n*rho/2. * 1.*(i+1)*(n-i-j-k+1)/(n+1)/n )
                
                if k > 0:
                    row.append( index_n(n,i,j,k) )
                    col.append( index_n(n+1,i+1,j,k-1) )
                    data.append( n*rho/2. * 1.*(i+1)*(n-i-j-k+1)/(n+1)/n )
                
                if i > 0:
                    row.append( index_n(n,i,j,k) )
                    col.append( index_n(n+1,i-1,j+1,k+1) )
                    data.append( n*rho/2. * 1.*(j+1)*(k+1)/(n+1)/n )
                
                if i+j+k+1 < n+1:
                    row.append( index_n(n,i,j,k) )
                    col.append( index_n(n+1,i,j+1,k+1) )
                    data.append( n*rho/2. * 1.*(j+1)*(k+1)/(n+1)/n )

                # outgoing
                row.append( index_n(n,i,j,k) )
                col.append( index_n(n+1,i+1,j,k) )
                data.append( -n*rho/2. * 1.*(i+1)*(n-i-j-k)/(n+1)/n )

                row.append( index_n(n,i,j,k) )
                col.append( index_n(n+1,i,j+1,k) )
                data.append( -n*rho/2. * 1.*(j+1)*(k)/(n+1)/n )

                row.append( index_n(n,i,j,k) )
                col.append( index_n(n+1,i,j,k+1) )
                data.append( -n*rho/2. * 1.*(j)*(k+1)/(n+1)/n )

                row.append( index_n(n,i,j,k) )
                col.append( index_n(n+1,i,j,k) )
                data.append( -n*rho/2. * 1.*(i)*(n-i-j-k+1)/(n+1)/n )

    return csc_matrix((data,(row,col)),shape=(Rsize0,Rsize1))

def selection_additive_component(n):
    Ssize0 = (n+1)*(n+2)*(n+3)/6
    Ssize1 = (n+2)*(n+3)*(n+4)/6
    
    row = []
    col = []
    data = []
    for i in range(n+1):
        for j in range(n+1-i):
            for k in range(n+1-i-j):
                this_ind = index_n(n,i,j,k)
                if i > 0:
                    row.append(this_ind)
                    col.append(index_n(n+1,i+1,j,k))
                    data.append( - 1./(n+1) * (i+1)*(n-i-j) )
                if j > 0:
                    row.append(this_ind)
                    col.append(index_n(n+1,i,j+1,k))
                    data.append( - 1./(n+1) * (j+1)*(n-i-j) )
                if k > 0:
                    row.append(this_ind)
                    col.append(index_n(n+1,i,j,k+1))
                    data.append( 1./(n+1) * (i+j)*(k+1) )
                if n-i-j-k > 0:
                    row.append(this_ind)
                    col.append(index_n(n+1,i,j,k))
                    data.append( 1./(n+1) * (i+j)*(n-i-j-k+1) )
    return csc_matrix((data,(row,col)), shape=(Ssize0,Ssize1))

def selection_dominance_component(n):
    Ssize0 = (n+1)*(n+2)*(n+3)/6
    Ssize2 = (n+3)*(n+4)*(n+5)/6
    
    row = []
    col = []
    data = []
    for i in range(n+1):
        for j in range(n+1-i):
            for k in range(n+1-i-j):
                this_ind = index_n(n,i,j,k)
                if i > 0 and k > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i+1,j,k+1))
                    data.append( 1./(n+1)/(n+2) * (i+1)*(k+1)*(i+j) )
                if i > 0 and n-i-j-k > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i+1,j,k))
                    data.append( 1./(n+1)/(n+2) * (i+1)*(n-i-j-k+1)*(i+j) )
                if j > 0 and k > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i,j+1,k+1))
                    data.append( 1./(n+1)/(n+2) * (j+1)*(k+1)*(i+j) )
                if j > 0 and n-i-j-k > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i,j+1,k))
                    data.append( 1./(n+1)/(n+2) * (j+1)*(n-i-j-k+1)*(i+j) )
                if i > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i+2,j,k))
                    data.append( - 1./(n+1)/(n+2) * (i+2)*(i+1)*(n-i-j) )
                if i > 0 and j > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i+1,j+1,k))
                    data.append( - 2./(n+1)/(n+2) * (i+1)*(j+1)*(n-i-j) )
                if j > 0:
                    row.append(this_ind)
                    col.append(index_n(n+2,i,j+2,k))
                    data.append( - 1./(n+1)/(n+2) * (j+2)*(j+1)*(n-i-j) )

    return csc_matrix((data,(row,col)), shape=(Ssize0,Ssize2))

