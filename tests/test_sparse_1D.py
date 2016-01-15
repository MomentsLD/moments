# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn
import sys
sys.path[:0] = ['../'] # pour l'import des modules dns le dossier parent...

import integration as it
import integration_multiD_sparse as ids
import utils as ut
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
gamma = -1 # same as in dadi
s = gamma/N
# dominance
h = 0.1
# mutation rate
theta = 1.0
# population sample size
m = 10
# simulation final time (number of generations)
tp = 100 # same as in dadi
# time step for the integration
dt = 0.01*tp


#-------------
# Our code   :
#-------------
f = lambda x: [N]
# sparse version
v = np.zeros(m+1)
start_time = time.time()
#v2sps = ids.integrate_N_cst(v, [N], [m], tp, dt, theta=theta, h=[h], gamma=[gamma], m=[0])
v2sps = ids.integrate_N_lambda_CN(v, f, [m], tp, dt, theta=theta, h=[h], gamma=[gamma], m=[0])
interval = time.time() - start_time
print('Total time sparse:', interval)


f = lambda x: N
v = np.zeros(m-1)
start_time = time.time()
#v2d = it.integrate_N_cst(v, N, m, tp, dt, theta=theta, h=h, gamma=gamma)
v2d = it.integrate_N_lambda(v, f, m, tp, dt, theta=theta, h=h, gamma=gamma)
interval = time.time() - start_time
print('Total time dense:', interval)

print((v2d-v2sps[1:m]<10**(-5)).all())

#print(v2sps[1:m])
#print(v2d)

# cas N = constant avec
# N = 1
# gamma = -1
# h = 0.1
# theta = 1.0
# tp = 100
# dt = 0.01*tp

tps_dense = [0.00055, 0.000779, 0.00125, 0.001196, 0.001406, 0.00273, 0.0071, 0.03365, 0.21339, 1.13958, 12.6997, 89.2014, 688.0574]
tps_sparse = [0.0092, 0.00727, 0.0097, 0.01045, 0.01540, 0.02519, 0.0401, 0.0879, 0.1746, 0.3334, 0.8636, 1.60537, 3.47627, 17.8928, 202.74965]
n = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
n2 = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 100000, 1000000]

# cas N = lambda fonction avec
# f = 1
# gamma = -1
# h = 0.1
# theta = 1.0
# tp = 100
# dt = 0.01*tp

tps_dense = [0.0038, 0.00575, 0.00438, 0.0076, 0.0140, 0.0296, 0.08371, 0.41619, 3.1728, 20.3298, 210.126, 1242.365]
tps_sparse = [0.156, 0.1522, 0.1550, 0.1665, 0.1681, 0.1866, 0.22287, 0.3260, 0.4173, 0.68159, 1.42691, 2.8669, 29.65, 311.70]
n = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
n2 = [5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000, 1000000]

plt.loglog(n, tps_dense, 'r')
plt.loglog(n2, tps_sparse, 'g')
plt.xlabel("degrees of freedom (sample size)")
plt.ylabel("time (s)")
#plt.title("convergences comparison (N = 1, n = 50): dadi vs our code")
plt.show()