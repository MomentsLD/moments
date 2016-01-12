import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

import jackknife as jk
import integration_multiD as itd
import integration as it
import utils as ut


def calc_error(sfs, ref):
    sfs = np.array(sfs)
    ref = np.array(ref)
    return np.amax(abs(sfs-ref)/ref)


Npop = np.array([10000, 10000])
n1 = 10
n2 = 10
nsample = np.array([n1, n2])
dims = nsample + np.ones(len(nsample))
d = int(np.prod(dims))
u = 1/4.0/Npop[0]

Tmax = 100000000
dt = 0.01*Tmax
# initialization
v = np.zeros([dims[0],dims[1]])
#v = np.eye(6)
#print(v)

start_time = time.time()
# matrix for mutations
B = itd.calcB(u, dims)
#print(B)

# matrix for drift
vd = itd.calcD(dims)
D = 1/4.0/Npop[0]*vd[0]
for i in range(1, len(Npop)):
    D = D + 1/4.0/Npop[i]*vd[i]


# matrix for selection
gamma = 1.0
hh = 0.5
s = gamma/Npop[0]*np.ones(2)
h = hh*np.ones(2)

S = itd.calcS(dims, s, h)
Sbis = itd.calcS_jk3(dims, s, h)
S2 = itd.calcS2(dims, s, h)
S2bis = itd.calcS2_jk3(dims, s, h)

#print(i," : ",Sbis[:,i])
# matrix for migration
m = 0.0/4.0/Npop[0]*np.array([[1, 1.7],[1, 1]])
Mi = itd.calcM(dims, m)
Mi2 = itd.calcM_jk3(dims, m)
'''for i in range(d):
    print(i," : ",sum(Mi2[:,i]))'''

Q = np.eye(d)-dt*(D+Sbis+S2+Mi)
#M = np.linalg.inv(Q)
# time loop
t=0.0
# all in 1D for the time integration...
v1 = v.reshape(d)
B1 = B.reshape(d)
J = jk.calcJK13(n1)
J2 = jk.calcJK23(n1)
#print(J2)

totmut = 0
'''while t<Tmax:
    #v1 = np.dot(M,(v1+dt*B1))
    v1 = np.linalg.solve(Q,v1+dt*B1)
    t += dt
    totmut += dt*sum(B1)
v = v1.reshape(dims)
test = v[0,:]
interval = time.time() - start_time
print('Total time:', interval)'''

start_time = time.time()
vv = itd.integrate_N_cst(np.zeros([dims[0],dims[1]]), Npop, nsample, Tmax/2.0/Npop[0], dt/2.0/Npop[0], gamma=gamma*np.ones(2), m=m, h=hh*np.ones(2))
interval = time.time() - start_time
print('Total time 2 :', interval)

#print("somme sur v1 : ",sum(v1))
#print("mutations totales",totmut)
#v2 = np.array([round(x,2) for x in v1])

#print(test)
#test 1D
# Initialisation
v1D = np.zeros(n1-1)
v1D = it.integrate_N_cst(v1D, 1, n1, Tmax/2.0/Npop[0], dt/2.0/Npop[0], gamma=gamma, h=hh)

print('erreur : ',calc_error(v1D,vv[0,1:n1]))#plt.plot(test[1:-1], 'g')
plt.plot(v1D, 'r')
plt.plot(vv[0,1:n1], 'g')
#plt.show()
#print("comp: ",v1D-vv[0,1:n1])

print(v1D)
print(vv[0,1:n1])

'''sf = ut.project_2D(vv, 5, 5)
v[0,0] = 0
v[n1,n2] = 0'''

'''vv[0,0] = 0
vv[n1,n2] = 0
vv[n1,0] = 0
vv[0,n2] = 0
plt.imshow(vv,interpolation='nearest')
plt.gca().invert_yaxis()
cbar = plt.colorbar()
plt.show()'''

