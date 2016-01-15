import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn
import sys
sys.path[:0] = ['../'] # pour l'import des modules dns le dossier parent...

import integration_multiD as itd
import integration_multiD_sparse as ims

def test_eq(A, B):
    A = np.array(A)
    B = np.array(B)
    return (abs(A-B)<10**(-7)).all()
dims = [4,4]

s = 1.0*np.ones(2)
h = 0.1*np.ones(2)
m = 1.0*np.ones((2,2))

f = lambda x: [1, 1]
n = 10
#print(n**2)
k = [n, n]
#v0 = np.zeros([n+1,n+1])
v0 = np.zeros(k+np.ones(2))
start_time = time.time()
v = ims.integrate_N_cst(v0, [1,1], k, 100, 1.0, gamma=1.0*np.ones(2), m=m, h=h)
#v = ims.integrate_N_lambda_CN(v0, f, k, 100, 1.0, gamma=1.0*np.ones(2), m=m, h=h)
interval = time.time() - start_time
print('Total time sparse:', interval)
#print(v[3])

start_time = time.time()
vv = itd.integrate_N_cst(v0, [1,1], [n,n], 100, 1.0, gamma=1.0*np.ones(2), m=m, h=h)
#vv = itd.integrate_N_lambda_CN(v0, f, k, 100, 1.0, gamma=1.0*np.ones(2), m=m, h=h)
interval = time.time() - start_time
print('Total time not sparse:', interval)
#print(vv[3])


print(test_eq(v,vv))

# comparaison performances sparse vs non sparse
# parametres
# f = lambda x: [1, 1]
# s = 1.0*np.ones(2)
# h = 0.1*np.ones(2)
# m = 1.0*np.ones((2,2))
nb_pts = [25, 100, 225, 400, 625, 900, 2500, 4900, 10000, 100000]
tps_sps = [0.2058, 0.310, 0.5013, 0.9680, 1.2267, 1.8385, 5.9256, 13.30243, 29.057, 406.34]
tps_nsps = [0.0461, 0.1687, 0.4732, 1.3342, 3.998, 7.855, 58.7868, 340.423]
plt.loglog(nb_pts, tps_sps, 'r')
plt.loglog(nb_pts[0:8],tps_nsps, 'g')
plt.show()
