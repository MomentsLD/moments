import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import math

#-----------------------------------
# 1 dimension case
# drift, mutations and selection (h=0.5)
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
h = 0.5
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


# Compute the order 2 Jackknife extrapolation coefficients
def calcJK(n):
    J = np.zeros((n, n-1))
    for i in range(1, n-1):
        J[i,i] = (1+n)*(2-(i+1)+n)/(2+n)/(3+n)
        J[i,i-1] = (i+2)*(n+1)/(n+2)/(n+3)
    J[0, 0] = (1+n)*(5+n)/(n+2)/(n+3)
    J[0, 1] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-3] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-2] = (1+n)*(5+n)/(n+2)/(n+3)
    return J

# Compute the selection linear system matrix
def calcS(n):
    S = np.zeros((n-1, n))
    for i in range(0, n-1):
        S[i, i] = s*h*(i+1)*(n-i)/(n+1)
        S[i, i+1] = -s*h/(n+1)*(n-i-1)*(i+2)
    return S
#------------


# Initialisation
v = np.zeros(n-1)
A = calcA(n)
B = calcB(n)

J = calcJK(n)
S = calcS(n)
Sbis = np.dot(S, J)

# We compute the linear system matrix:
Q = np.eye(n-1)-dt*(1/(4*N)*A+Sbis)
M = np.linalg.inv(Q)

STS=-np.dot(np.linalg.inv(1/(4*N)*A+Sbis),B)
# Time loop to solve the system
t = 0.0
while t < Tmax:
    # Implicit Euler scheme
    v = np.dot(M, (v+dt*B))
    # Jackknife estimation of the Phi n+1
    #v += dt*np.dot(Sbis, v)
    t += dt

#print(v)

'''dadi = np.array([1.0137582574409385, 0.5139075013164367, 0.34739174219289776,
                 0.26421192622217937, 0.21436810850794916, 0.18119367896963523,
                 0.15754583843814976, 0.13985321770950876, 0.12613175058654902,
                 0.11519102311751207, 0.10627352327042397, 0.09887426578962252,
                 0.09264366713689734, 0.08733204547545896, 0.08275632067725357,
                 0.07877920184389683, 0.07529572042432635, 0.07222425230868665,
                 0.0695003750655469, 0.067072568022725, 0.06489914091094717,
                 0.0629460001644118, 0.06118499793961786, 0.05959269389337837,
                 0.05814941414785614, 0.05683852743113732, 0.055645882088975925,
                 0.05455936375031763, 0.05356854452367438, 0.052664402367549164,
                 0.05183909478955281, 0.05108577499020745, 0.05039844144841312,
                 0.049771814063924676, 0.049201231545832926, 0.04868256591627893,
                 0.04821215089178263, 0.047786721586181975, 0.04740336350350011,
                 0.047059469195405684, 0.04675270127508287, 0.046480960728516894,
                 0.046242359661235834, 0.046035197775281166, 0.04585794199656609,
                 0.04570920877363494, 0.0455877486503791, 0.045492432781518814,
                 0.0454222411137445])'''

X = np.arange(1,n)
#plt.plot(X, abs(dadi/dadi[0]-v/v[0])*dadi[0]/dadi, 'r')
#plt.plot(X, abs(dadi/dadi[0]-v/v[0])*dadi[0]/dadi, 'r')
#plt.plot(X, abs(dadi/dadi[0]-dadi10/dadi10[0])*dadi[0]/dadi)
#plt.plot(X, dadi/dadi[0])
plt.plot(X, v/v[0], 'r')

plt.show()




