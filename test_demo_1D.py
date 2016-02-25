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
nu = 1.0
params = (nu, T)
#fs = dadi.Demographics1D.snm([], (ns,), pts)
fs = dadi.Demographics1D.two_epoch(params, (ns,), pts)
print(fs)

#fs2 = moments.Demographics1D.snm([ns])
fs2 = moments.Demographics1D.two_epoch(params, [ns])
#fs2.unmask_all()

print(fs2)
print('error : ', stats.entropy(fs[1:ns], fs2[1:ns]))