# -*- coding: UTF-8 -*-
import numpy as np
import time
import scipy.stats as stats

import moments
import dadi

#-------------
# Parameters :
#-------------

n1 = 20
n2 = 25
pts = 100
s = 0.25
T = 1.0
Ts = 0.1
nuB = 1.1
nuF = 3.0
nuPre = 1.0
TPre = 10
m = 1.0
m2 = 2.0

#params = (nuB,nuF,m,T,Ts)
#params = (nuB,nuF,T,Ts)
#params = (nuB,nuF,T,m)
#params = (s,nuB,nuF,T,m,m2)
params = (nuPre,TPre,s,nuB,nuF,T,m,m2)
# dadi
start_time = time.time()
#fs = dadi.Demographics2D.bottlegrowth_split(params, (n1,n2,), pts)
fs = dadi.Demographics2D.IM_pre(params, (n1,n2,), pts)
interval = time.time() - start_time
#print('Total time dadi:', interval)
#print(fs[0,:])


# moments
start_time = time.time()
#fs2 = moments.Demographics2D.bottlegrowth_split(params, [n1, n2])
fs2 = moments.Demographics2D.IM_pre(params, [n1, n2])
interval = time.time() - start_time
print('Total time moments:', interval)
print(fs2[0,0:8])
#print(fs2)

print('error : ', stats.entropy(fs.reshape((n1+1)*(n2+1))[1:-1], fs2.reshape((n1+1)*(n2+1))[1:-1]))

'''import pylab
moments.Plotting.plot_single_2d_sfs(fs2)
pylab.show()
moments.Plotting.plot_single_2d_sfs(fs)
pylab.show()'''