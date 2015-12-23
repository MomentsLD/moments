import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import jackknife as jk
import integration_multiD as itd
import integration as it

Npop = np.array([10000, 10000])
n1 = 50
n2 = 50
nsample = np.array([n1, n2])
dims = nsample + np.ones(len(nsample))
d = int(np.prod(dims))
u = 1/4.0/Npop[0]

Tmax = 1000000
dt = 0.01*Tmax
# initialization
v = np.zeros([dims[0],dims[1]])
#v = np.eye(6)
#print(v)

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
gamma = 1.0
hh = .1
s = gamma/Npop[0]*np.ones(2)
h = hh*np.ones(2)

S = itd.calcS(dims, s, h)
S2 = itd.calcS2(dims, s, h)
'''for i in range(d):
    print(i," : ",sum(S2[:,i]))'''
# matrice for migration
m = 1/4.0/Npop[0]*np.array([[1, 2],[2, 1]])
Mi = itd.calcM(dims, m)
'''for i in range(d):
    print(i," : ",sum(Mi[:,i]))'''

#Q = np.eye(d)-dt*(D+S+Mi)
Q = np.eye(d)-dt*(D+S+S2)
M = np.linalg.inv(Q)
# time loop
t=0.0
# all in 1D for the time integration...
v1 = v.reshape(d)
B1 = B.reshape(d)
J = jk.calcJK12(n1)
J2 = np.dot(jk.calcJK12(n1+1),J)
#print(J2)
#print(J2[n1-1,n1-3],", et ",J2[n1-1,n1-2])
totmut = 0
while t<Tmax:
    v1 = np.dot(M,(v1+dt*B1))
    t += dt
    totmut += dt*sum(B1)
#print("somme sur v1 : ",sum(v1))
#print("mutations totales",totmut)
#v2 = np.array([round(x,2) for x in v1])
v = v1.reshape(dims)
test = v[0,:]

#test 1D
# Initialisation
v = np.zeros(n1-1)
v = it.integrate_N_cst(v, 10000, n1, 50, dt, gamma=gamma, h=hh)

print(test[1:-1])
plt.plot(test[1:-1], 'g')
plt.plot(v, 'r')
plt.show()
'''
nrows, ncols = n1+1, n2+1
plt.imshow(v)#,interpolation='nearest', cmap=cm.gist_rainbow)
plt.show()'''
