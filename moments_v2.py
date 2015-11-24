import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import math

#-----------------------------------
# 1 dimension case
# drift, mutations and selection
# constant N (total pop size)
#-----------------------------------

#-------------
# Parameters :
#-------------
# total population size
N = 10000
# selection
gamma = 1 # same as in dadi
s = gamma/N
# dominance
h = 1
# mutation rate
u = 1/(4*N) # same as in dadi default
# population sample size
n = 50
# simulation final time (number of generations)
t = 100.0 # same as in dadi
Tmax = t*2*N
# time step for the integration
dt = 100
#-------------

#------------
# Functions :
#------------

# Compute the matrix A
def calcA(n):
    A = np.zeros((n-1, n-1))
    for i in range(0, n-1):
        A[i,i] = -2*(i+1)*(n-(i+1))
        if (i < n-2):
            A[i, i+1] = (n-(i+1)-1)*(i+2)
        if (i > 0):
            A[i, i-1] = (n-(i+1)+1)*i
    return A

# Compute the vector B
def calcB(n):
    B = np.zeros(n-1)
    for i in range(0, n-1):
        B[i] = u*10**(math.log(misc.comb(n, i), 10)-(i-1)*math.log(2.0*N, 10)+(n-i)*math.log(1-1/(2*N), 10))
    return B

# Compute the linear system matrix M=(I-dt/(4N)*A)**(-1)
def calcM(A):
    Q = np.eye(n-1)-dt/(4*N)*A
    M = np.linalg.inv(Q)
    return M

# Steady state
def solstat(A, B):
    return -4*N*np.dot(np.linalg.inv(A), B)


# Compute the order 2 Jackknife extrapolation coefficients for 1 jump (phi_n -> phi_(n+1))
def calcJK1(n):
    J = np.zeros((n, n-1))
    for i in range(1, n-1):
        J[i,i] = (1+n)*(2-(i+1)+n)/(2+n)/(3+n)
        J[i,i-1] = (i+2)*(n+1)/(n+2)/(n+3)
    J[0, 0] = (1+n)*(5+n)/(n+2)/(n+3)
    J[0, 1] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-3] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-2] = (1+n)*(5+n)/(n+2)/(n+3)
    return J

# Compute the order 2 Jackknife extrapolation coefficients for 2 jumps (phi_n -> phi_(n+2))


# Compute the selection linear system matrix for n+1 order terms
def calcS1(n):
    S = np.zeros((n-1, n))
    for i in range(0, n-1):
        S[i, i] = s*h*(i+1)*(n-i)/(n+1)
        S[i, i+1] = -s*h/(n+1)*(n-i-1)*(i+2)
    return S

# Compute the selection linear system matrix for n+2 order terms
def calcS2(n):
    S = np.zeros((n-1, n+1))
    for i in range(0, n-1):
        S[i, i+1] = s*(1-2*h)*(i+2)/(n+1)/(n+2)*(i+1)*(n-i)
        S[i, i+2] = -s*(1-2*h)*(i+2)/(n+1)/(n+2)*(n-i-1)*(i+3)
    return S
#------------


# Initialisation
v = np.zeros(n-1)
A = calcA(n)
B = calcB(n)

# Matrix for the selection, JK n->n+1
J = calcJK1(n)
S = calcS1(n)
Sbis = np.dot(S, J)

# Matrix for the part h!=0.5, JK n->n+2
S2 = calcS2(n)
J2 = np.dot(calcJK1(n+1),J)
Sbis2 = np.dot(S2, J2)

# We compute the total linear system matrix:
Q = np.eye(n-1)-dt*(1/(4*N)*A+Sbis+Sbis2)
M = np.linalg.inv(Q)

STS=-np.dot(np.linalg.inv(1/(4*N)*A+Sbis+Sbis2),B)

# Time loop to solve the system
t = 0.0
while t < Tmax:
    # Implicit Euler scheme
    v = np.dot(M, (v+dt*B))
    t += dt

#print(v)

dadi = np.array([1.0379895381487605, 0.5386025736421945, 0.3725457500391616,
                 0.28981881710591567, 0.24042058645857584, 0.2076831597117348,
                 0.18446240718064644, 0.16718558986287718, 0.15386723572719163,
                 0.1433154925137447, 0.13477138127748248, 0.12772842541170018,
                 0.12183553004241351, 0.11684148673753889, 0.11256167849477533,
                 0.1088572724794649, 0.10562175857421859, 0.10277197709626111,
                 0.1002419818238874, 0.09797874600456072, 0.09593909703424562,
                 0.09408748887066741, 0.09239435720892503, 0.09083488742607815,
                 0.08938807968709908, 0.08803603116433786, 0.08676337902909402,
                 0.08555686396044573, 0.08440498501080387, 0.08329772443391414,
                 0.08222632659288785, 0.08118311902789307, 0.08016136664511178,
                 0.07915515210821633, 0.07815927708882844, 0.07716918021450067,
                 0.07618086844802208, 0.07519085931583466, 0.07419613193027942,
                 0.0731940851595783, 0.07218250161965667, 0.07115951641433417,
                 0.07012358975084354, 0.06907348271790477, 0.0680082356426141,
                 0.06692714854699085, 0.0658297633103498, 0.06471584721372463,
                 0.06358537760044833])

X = np.arange(1,n)
X2 = np.arange(1,n+2)
#plt.plot(X, abs(dadi/dadi[0]-v/v[0])*dadi[0]/dadi, 'r')
plt.plot(X, 1/X, 'g')
plt.plot(X, dadi/dadi[0])
plt.plot(X, v/v[0], 'r')
#plt.plot(X, STS/STS[0])
#plt.plot(X2, 1/X2)
#X3 = np.dot(J2, 1/X)
#plt.plot(X2, X3)

plt.show()




