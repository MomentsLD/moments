# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate
import time

import dadi

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
N = 1
# selection
gamma = 2 # same as in dadi
s = gamma/N
# dominance
h = 0.5
# mutation rate
theta = 1.0
# population sample size
n = 100
# simulation final time (number of generations)
tp = 100 # same as in dadi
# time step for the integration
dt = 0.01*tp
#-------------
# Our code   :
#-------------
v = np.zeros(n-1)
#v2 = np.random.rand(n-1)
start_time = time.time()
v = it.integrate_N_cst(v, N, n, tp, dt, theta=theta, h=h, gamma=gamma)
interval = time.time() - start_time
print('Total time our code:', interval)
#---------
# Dadi   :
#---------
# use ∂a∂i to simulate allele frequency spectrum
def model((nu, t), (n1, ), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx,gamma=gamma,h=h)
    phi = dadi.Integration.one_pop(phi, xx, t, nu=nu, gamma=gamma, h=h)
    sfs = dadi.Spectrum.from_phi(phi, (n1, ), (xx, ))
    return sfs

start_time = time.time()
fs = model((1, tp), (n, ), 150)
interval = time.time() - start_time
print('Total time dadi:', interval)
# define the plotting environment
fig = plt.figure()
fig1 = fig.add_subplot(211)
fig2 = fig.add_subplot(212)

X = np.arange(1,n)
#plt.plot(X, abs(1.0/X-v)*X, 'g')
#plt.plot(X, abs(1.0/X-fs[1:n])*X, 'r')
fig1.plot(X, fs[1:n], 'r')
fig1.plot(X, v, 'g')

fig2.plot(X, abs(v-fs[1:n])/fs[1:n], 'r')
fig2.set_yscale('log')
#plt.xlabel("frequency in the popuation")
#plt.ylabel("relative error (%)")
#plt.title("2 jumps extrapolation for 1/x")
plt.show()




