import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import integration_multiD as itd

Npop = np.array([10000, 10000])
n1 = 5
n2 = 5
dims = np.array([n1+1, n2+1])
d = np.prod(dims)

Tmax = 100000
dt = 0.01*Tmax
# initialization
v = np.zeros([dims[0],dims[1]])

# matrice for mutations
B = itd.calcB(1/4/Npop[0], dims)
#print(B)
vd = itd.calcD(dims)
# matrice for drift
D = 1/4/Npop[0]*vd[0]
for i in range(1, len(Npop)):
    D = D + 1/4/Npop[i]*vd[i]
#print(D)
ii = itd.index_1D([1,1], dims)
print(D[d-1,:])
print(D[:,d-1])
ii1 = itd.index_1D([0,1], dims)
ii2 = itd.index_1D([1,0], dims)
#print(ii1, ii2)
Q = np.eye(d)-dt*D
M = np.linalg.inv(Q)
# time loop
t=0.0
# all in 1D for the time integration...
v1 = v.reshape(d)
B1 = B.reshape(d)

while t<Tmax:
    v1 = np.dot(M,(v1+dt*B1))
    t += dt
v2 = np.array([round(x,2) for x in v1])
v = v2.reshape(dims)
print(v)
print((n1-1)/4/Npop[0])
'''v[0,0] = 0
v[0,n2] = 0
v[n1,0] = 0
v[n1,n2] = 0
nrows, ncols = n1+1, n2+1
plt.imshow(v)#,interpolation='nearest', cmap=cm.gist_rainbow)
plt.show()'''
