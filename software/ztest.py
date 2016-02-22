# -*- coding: UTF-8 -*-
import numpy as np
import time

import moments


#-------------
# Parameters :
#-------------
'''
m = np.array([[1, 5],[10, 1]])
S = moments.Integration.calcM([25, 30], m)

d = S[0].todense()

np.savetxt('migration_matrix.csv', d, delimiter=',')'''
dim = 3
N = 1
gamma = [0, 0.5, -2]
h = [0.5, 0.1, 0.9]
theta = 1.0
n = 15
n2 = 20
n3 = 18
tp = 10
dt = 0.005*tp
#mig = np.ones([dim, dim])
mig = np.array([[0, 5, 2],[1, 0, 1],[10, 0, 1]])
f = lambda x: (dim-1)*[N]+[N+0.0001*x]

#sfs = np.zeros(dim*[n+1])
sfs = moments.Spectrum(np.zeros([n+1, n2+1, n3+1]))

#sfs.unmask_all()
#print(sfs)
start_time = time.time()

#sfs = moments.Integration.integrate(sfs, f, dim*[n], tp, dt, theta=1.0, h=dim*[h], gamma=dim*[gamma], m=mig)
sfs.integrate(f, [n, n2, n3], tp, dt, theta=1.0, h=dim*h, gamma=gamma, m=mig)

interval = time.time() - start_time
print('Total time our code:', interval)
sfs.to_file('1_pop.fs')
#print(sfs)
#sfs2 = sfs.project(dim*[10])
#print(sfs[:,0,0,0])
