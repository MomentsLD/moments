import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import jackknife as jk
import integration_multiD as itd

Npop = np.array([10000, 10000])
n1 = 5
n2 = 5
nsample = np.array([n1, n2])
dims = nsample + np.ones(len(nsample))
d = int(np.prod(dims))
u = 1/2.0/Npop[0]

Tmax = 1000000
dt = 0.01*Tmax
# initialization
v = np.zeros([dims[0],dims[1]])

# matrice for mutations
B = itd.calcB(u, dims)
#print(B)

# matrice for drift
vd = itd.calcD(dims)
D = 1/4.0/Npop[0]*vd[0]
for i in range(1, len(Npop)):
    D = D + 1/4.0/Npop[i]*vd[i]

'''for i in range(d):
#print(D[:,i])
    print(itd.index_nD(i,dims),", ",sum(D[:,i]))
    if sum(D[:,i]) != 0: print(D[:,i])'''


# matrice for selection
s = -1/4.0/Npop[0]*np.ones(2)
h = 1/2.0*np.ones(2)

S2 = itd.calcS2(dims, s, h)
'''for i in range(d):
    print(i," : ",sum(S2[:,i]))'''

#Q = np.eye(d)-dt*D
Q = np.eye(d)-dt*(D+S2)
M = np.linalg.inv(Q)
# time loop
t=0.0
# all in 1D for the time integration...
v1 = v.reshape(d)
B1 = B.reshape(d)
J = jk.calcJK12(n1)
#print(J)
totmut = 0
while t<Tmax:
    v1 = np.dot(M,(v1+dt*B1))
    t += dt
    totmut += dt*sum(B1)
print("somme sur v1 : ",sum(v1))
print("mutations totales",totmut)
v2 = np.array([round(x,2) for x in v1])
v = v2.reshape(dims)
print(v)

'''
nrows, ncols = n1+1, n2+1
plt.imshow(v)#,interpolation='nearest', cmap=cm.gist_rainbow)
plt.show()'''
