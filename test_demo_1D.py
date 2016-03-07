# -*- coding: UTF-8 -*-
import numpy as np
import time
import scipy.stats as stats

import moments
import dadi

#-------------
# Parameters :
#-------------

ns = 20
pts = 100
T = 10
Tf = 5
nu = 1.5
nuF = 2.0
params = (nu, nuF, T, Tf)
#fs = dadi.Demographics1D.snm([], (ns,), pts)
fs = dadi.Demographics1D.three_epoch(params, (ns,), pts)
print(fs)

#fs2 = moments.Demographics1D.snm([ns])
fs2 = moments.Demographics1D.three_epoch(params, [ns])
#fs2.unmask_all()

print(fs2)
print('error : ', stats.entropy(fs[1:ns], fs2[1:ns]))