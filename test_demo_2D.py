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
T = 10
nu = 1.0
params = (nu, T)
fs = dadi.Demographics2D.snm(params, (n1,n2,), pts)
print(fs)

fs2 = moments.Demographics2D.snm([n1, n2])

print(fs2)
print('error : ', stats.entropy(fs.reshape((n1+1)*(n2+1))[1:-1], fs2.reshape((n1+1)*(n2+1))[1:-1]))