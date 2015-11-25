import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import math

import integration as it
import jackknife as jk
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
h = 0.1
# mutation rate
u = 1/(4*N) # same as in dadi default
# population sample size
n = 50
# simulation final time (number of generations)
tp = 100 # same as in dadi
Tmax = tp*2*N
# time step for the integration
dt = 100
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
#------------
f = lambda x: 10000+0.01*x
# Initialisation
v = np.zeros(n-1)
v = it.integrate_N_lambda(v, f, n, tp, dt, u=1, h=h)

B = it.calcB(u, n, N)
D = it.calcD(n)
S1 = it.calcS1(s, h , n)
S2 = it.calcS2(s, h , n)
ss = it.initialize(N, D, B, S1, S2)
X = np.arange(1,n)

#plt.plot(X, 1/X, 'g')
plt.plot(X, ss)
#plt.plot(X, abs(ss-v)/ss, 'g')
#plt.plot(X, dadi/dadi[0])
plt.plot(X, v, 'r')
#plt.yscale('log')
plt.xlabel("frequency in the popuation")
plt.ylabel("relative error (%)")
#plt.title("2 jumps extrapolation for 1/x")
plt.show()




