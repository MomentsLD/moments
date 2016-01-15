# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import sys
sys.path[:0] = ['../']
#import seaborn

import dadi

import utils as ut
import integration as it
import integration_multiD as itd
#-----------------------------------
# 1 dimension case
# drift, mutations and selection
#-----------------------------------

def calc_error(sfs, ref):
    sfs = np.array(sfs)
    ref = np.array(ref)
    return np.amax(abs(sfs-ref)/ref)

#-------------
# Parameters :
#-------------
# total population size
N = 1
# selection
gamma = 1.0 # same as in dadi
# dominance
h = 0.1
# mutation rate
theta = 1.0
#migration rates
m12 = 5.0
m21 = 10.0


Npop = np.array([1, 1])
m1 = 50
m2 = 50
n1 = 20
n2 = 20
nsample = np.array([n1, n2])
dims = nsample + np.ones(len(nsample))
d = int(np.prod(dims))
u = 1/4.0/Npop[0]
m = np.array([[1, m12],[m21, 1]])
tp = 100 # same as in dadi
# time step for the integration
dt = 0.01*tp

#v1D = np.zeros(n1+n2-1)
#v1D = it.integrate_N_cst(v1D, N, n1+n2, tp, dt, gamma=-1)
#v2D = ut.split_pop_12(v1D, n1, n2)

#-------------
# Our code   :
#-------------
f = lambda x: [N, N]#+2*x/1000.0

#vv = np.zeros([dims[0],dims[1]])
vv = np.zeros([m1+1,m2+1])
start_time = time.time()
msample = np.array([m1, m2])
#vv = itd.integrate_N_cst(vv, Npop, msample, tp, dt, gamma=gamma*np.ones(2), m=m, h=h*np.ones(2))
vv = itd.integrate_N_lambda_CN(vv, f, msample, tp, dt, gamma=gamma*np.ones(2), m=m, h=h*np.ones(2))
vv = ut.project_2D(vv, n1, n2)
interval = time.time() - start_time
print('Total time our code:', interval)
#print("somme var : ",vv.sum())
vv[0,0] = 0
vv[n1,n2] = 0

#for i in range(n1):
#print(i,': ',vv[i,:])

#vv = np.ma.asarray(vv)
#vv.mask.flat[0] = vv.mask.flat[-1] = True
#print(vv)

print(vv[0,:])

#---------
# Dadi   :
#---------
# use ∂a∂i to simulate allele frequency spectrum
def model((nu1, nu2, t), (n1,n2), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, t, nu1, nu2, m12=m12, m21=m21, gamma1=gamma, gamma2=gamma, h1=h, h2=h)
    sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return sfs
start_time = time.time()
fs = model((1,1,tp), (n1,n2), 80)
interval = time.time() - start_time
print('Total time dadi:', interval)
#print(fs)
#fs[0,0] = 0
#fs[n1,n2] = 0

#for i in range(n1):
#print(i,': ',fs[i,:])
#print(fs[0,:])
#print('erreur : ',calc_error(fs, vv))
#for i in range(1,n1):
#    print('i = ',i,', ',vv[i,i],', ',fs[i,i])
'''#test 1D
N = 1
v1D = np.zeros(m1-1)
v1D = it.integrate_N_cst(v1D, N, m1, tp, dt, gamma=gamma, h=h)
v1D = ut.project_1D(v1D, n1)
print "test 1D"
print(v1D)

#dadi 1D
def model1D((nu, t), (n1, ), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx,gamma=gamma,h=h)
    phi = dadi.Integration.one_pop(phi, xx, t, nu=nu, gamma=gamma, h=h)
    sfs = dadi.Spectrum.from_phi(phi, (n1, ), (xx, ))
    return sfs
sfs1D = model1D((N, tp), (n1, ), 1000)
print "test dadi1D"
print(sfs1D)'''
'''for i in range(n1+1):
    for j in range(n2+1):
        if vv[i,j]<0.00001 and fs[i,j]<0.00001:
            vv[i,j] = 0.0000000000001
            fs[i,j] = 0.0000000000001
ec = abs(vv-fs)/fs
plt.imshow(ec,interpolation='nearest')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
plt.show()'''

'''plt.imshow(v2D,interpolation='nearest')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
plt.show()'''

plt.imshow(vv,interpolation='nearest')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
plt.show()

'''plt.imshow(fs,interpolation='nearest')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
plt.show()'''

