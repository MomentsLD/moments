import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import math

#-----------------------------------
# 1 dimension case
# drift, mutations and selection
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
tp = 100 # same as in dadi
Tmax = tp*2*N
# time step for the integration
dt = 1000
#-------------

#------------
# Functions :
#------------
arrT = np.array([0, 100000, 200000, 300000]) # in number of generations
arrN = np.array([10000, 20000, 10000]) # population size described as step fuctions corresponding to the times above
# For a non constant size population
def popsize(t, arrT, arrN):
    Nr = arrN[0]
    for i in range(0, 4-2):
        if (t >= arrT[i]) and (t < arrT[i+1]):
            Nr= arrN[i]
    return Nr



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
def calcB(n, Nref):
    B = np.zeros(n-1)
    for i in range(0, n-1):
        B[i] = u*10**(math.log(misc.comb(n, i), 10)-(i-1)*math.log(2.0*Nref, 10)+(n-i)*math.log(1-1/(2*Nref), 10))
    return B

# Steady state
def solstat(A, B, S1, S2):
    return -np.dot(np.linalg.inv(1/(4*N)*A+S1+S2), B)

# The choice i' in n samples that best approximates the frequency of \i/(n + 1) is i*n / (n + 1)
def index_bis(i,n):
    return int(max(round(i*n/(n+1)),2))

# Compute the order 2 Jackknife extrapolation coefficients for 1 jump (phi_n -> phi_(n+1))
def calcJK12(n):
    J = np.zeros((n, n-1))
    for i in range(1, n-1):
        J[i,i] = (1+n)*(2-(i+1)+n)/(2+n)/(3+n)
        J[i,i-1] = (i+2)*(n+1)/(n+2)/(n+3)
    J[0, 0] = (1+n)*(5+n)/(n+2)/(n+3)
    J[0, 1] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-3] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-2] = (1+n)*(5+n)/(n+2)/(n+3)
    return J

# Compute the order 3 Jackknife extrapolation coefficients for 1 jump (phi_n -> phi_(n+1))
def calcJK13(n):
    J = np.zeros((n, n-1))
    for i in range(0, n):
        ibis = index_bis(i+1,n)-1
        if (i == n-1): ibis -= 1
        #print(ibis)
        J[i,ibis] = -(1+n)*((2+i)*(2+n)*(-6-n+(i+1)*(3+n))-2*(4+n)*(-1+(i+1)*(2+n))*(ibis+1)+(12+7*n+n**2)*(ibis+1)**2)/(2+n)/(3+n)/(4+n) # beta
        J[i,ibis-1] = (1+n)*(4+(1+i)**2*(6+5*n+n**2)-(i+1)*(14+9*n+n**2)-(4+n)*(-5-n+2*(i+1)*(2+n))*(ibis+1)+(12+7*n+n**2)*(ibis+1)**2)/(2+n)/(3+n)/(4+n)/2 # alpha
        J[i,ibis+1] = (1+n)*((2+i)*(2+n)*(-2+(i+1)*(3+n))-(4+n)*(1+n+2*(i+1)*(2+n))*(ibis+1)+(12+7*n+n**2)*(ibis+1)**2)/(2+n)/(3+n)/(4+n)/2 # gamma
    return J

# Compute the order 2 Jackknife extrapolation coefficients for 2 jumps (phi_n -> phi_(n+2))
def calcJK22(n):
    J = np.zeros((n+1, n-1))
    for i in range(1, n-1):
        J[i,i] = (1+n)*(2-2*(i+1)+n)/(4+n)/(3+n)
        J[i,i-1] = 2*(i+2)*(n+1)/(n+4)/(n+3)
    J[0,0] = (8+9*n+n**2)/(12+7*n+n**2)
    J[0,1] = 4*(1+n)/(12+7*n+n**2)
    J[n-1,n-2] = 6*(1+n)/(12+7*n+n**2)
    J[n-1,n-3] = (-2-n+n**2)/(12+7*n+n**2)
    J[n,n-2] = (8+9*n+n**2)/(12+7*n+n**2)
    J[n,n-3] = 4*(1+n)/(12+7*n+n**2)
    return J

# Compute the order 3 Jackknife extrapolation coefficients for 2 jumps (phi_n -> phi_(n+2))
def calcJK23(n):
    J = np.zeros((n+1, n-1))
    for i in range(0, n+1):
        ibis = index_bis(i+1,n)-1
        if (i == n-1) or (i == n): ibis = n-3
        J[i,ibis] = -(1+n)*((2+i)*(2+n)*(-9-n+(i+1)*(3+n))-2*(5+n)*(-2+(i+1)*(2+n))*(ibis+1)+(20+9*n+n**2)*(ibis+1)**2)/(3+n)/(4+n)/(5+n)
        J[i,ibis-1] = (1+n)*(12+(1+i)**2*(6+5*n+n**2)-(i+1)*(22+13*n+n**2)-(5+n)*(-8-n+2*(i+1)*(2+n))*(ibis+1)+(20+9*n+n**2)*(ibis+1)**2)/(3+n)/(4+n)/(5+n)/2
        J[i,ibis+1] = (1+n)*((2+i)*(2+n)*(-4+(i+1)*(3+n))-(5+n)*(n+2*(i+1)*(2+n))*(ibis+1)+(20+9*n+n**2)*(ibis+1)**2)/(3+n)/(4+n)/(5+n)/2
    return J


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
# initial size of the total population
N0 = popsize(0, arrT, arrN)

# Mutation source term (initial)
B = calcB(n,N0)

# Matrix for the drift part
A = calcA(n)

# Matrix for the selection, JK n->n+1
J = calcJK13(n)
S = calcS1(n)
Sbis = np.dot(S, J)

# Matrix for the part h!=0.5, JK n->n+1->n+2
S2 = calcS2(n)
J2 = np.dot(calcJK13(n+1),J)
Sbis2 = np.dot(S2, J2)

# Initialisation
v = np.zeros(n-1)
#v = solstat(A, B, Sbis, Sbis2)

# We compute the total linear system matrix (initial)
Q = np.eye(n-1)-dt*(1/(4*N0)*A+Sbis+Sbis2)
M = np.linalg.inv(Q)

# Time loop to solve the system
t = 0.0
while t < Tmax:
    '''N = 10000#popsize(t, arrT, arrN)
    B = calcB(n,N)
    v += dt*B+dt*np.dot(1/(4*N)*A+Sbis+Sbis2,v)# Explicit Euler'''
    
    N = 10000#popsize(t, arrT, arrN)
    B = calcB(n,N)
    Q = np.eye(n-1)-dt*(1/(4*N)*A+Sbis+Sbis2)
    M = np.linalg.inv(Q)
    # Implicit Euler scheme
    v = np.dot(M, (v+dt*B))
    t += dt

#print(v)
ss = solstat(A, B, Sbis, Sbis2)
'''dadi = np.array([1.0379895381487605, 0.5386025736421945, 0.3725457500391616,
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
                 0.06358537760044833])'''

tstimp1000 = [5.47734345e-16, 1.05286405e-15, 7.61034798e-16, 1.34507163e-15,
            1.17920088e-15, 5.11896814e-16, 3.84219480e-16, 4.23920517e-16,
            6.90916059e-16, 7.41781239e-16, 6.57337028e-16, 6.93580117e-16,
            7.27124791e-16, 6.06561411e-16, 4.72217094e-16, 1.62761945e-16,
            6.70990475e-16, 5.17196824e-16, 1.76749951e-16, 5.42497873e-16,
            5.54031006e-16, 7.53245244e-16, 7.67048274e-16, 1.17032596e-15,
            1.18926686e-15, 1.81129673e-15, 2.45048690e-15, 2.48504359e-15,
            2.30904406e-15, 1.91433085e-15, 1.93927427e-15, 1.96419396e-15,
            1.54717878e-15, 1.45492873e-15, 1.02009238e-15, 8.03584702e-16,
            3.48861321e-16, 1.17818209e-16, 5.96988807e-16, 1.57342048e-15,
            2.20911334e-15, 2.73884280e-15, 2.77930332e-15, 3.20631403e-15,
            3.51705900e-15, 4.36806420e-15, 4.71002427e-15, 4.65420673e-15,
            5.01559707e-15]
tstexp10 = [2.34156433e-14, 4.19829538e-14, 5.85996794e-14, 7.31229850e-14,
            8.63764642e-14, 9.87960852e-14, 1.10463100e-13, 1.20817347e-13,
            1.30352830e-13, 1.39083982e-13, 1.47637896e-13, 1.55916810e-13,
            1.64039353e-13, 1.71960160e-13, 1.79599902e-13, 1.87338998e-13,
            1.95090481e-13, 2.02051559e-13, 2.08918442e-13, 2.15190823e-13,
            2.21612403e-13, 2.27668375e-13, 2.33757961e-13, 2.38941549e-13,
            2.44394340e-13, 2.49153928e-13, 2.54238016e-13, 2.59272881e-13,
            2.63860762e-13, 2.68006320e-13, 2.72144822e-13, 2.75860130e-13,
            2.78934232e-13, 2.81808502e-13, 2.84719119e-13, 2.87798121e-13,
            2.91066629e-13, 2.93838614e-13, 2.96942233e-13, 3.00039183e-13,
            3.03384899e-13, 3.06874886e-13, 3.10018652e-13, 3.12936249e-13,
            3.16274787e-13, 3.19265783e-13, 3.22031088e-13, 3.24562475e-13,
            3.27546353e-13]


X = np.arange(1,n)
'''JK23 = np.dot(calcJK13(n+1),calcJK13(n))
JK2 = calcJK2(n)
JK3 = calcJK23(n)
Y1 = np.dot(J2,1/X)
X2 = np.arange(1,n+2)
Y = np.dot(JK23, 1/X)
Y2 = np.dot(JK2, 1/X)
Y3 = np.dot(JK3, 1/X)'''
'''plt.plot(X2, Y, 'g')
plt.plot(X2, 1/X2)
plt.plot(X2, Y1,'r')'''
'''plt.plot(X2, abs(Y1-1/X2)*X2)
plt.plot(X2, abs(Y-1/X2)*X2, 'g')
plt.plot(X2, abs(Y2-1/X2)*X2, 'r')
plt.plot(X2, abs(Y3-1/X2)*X2, 'y')'''
#plt.plot(X, 1/X, 'g')
plt.plot(X, ss)
#plt.plot(X, abs(ss-v)/ss, 'g')
#plt.plot(X, dadi/dadi[0])
plt.plot(X, v, 'r')
#plt.plot(X, tstexp10)
#plt.plot(X, tstimp1000, 'g')
#plt.yscale('log')
plt.xlabel("frequency in the popuation")
plt.ylabel("relative error (%)")
#plt.title("2 jumps extrapolation for 1/x")
plt.show()




