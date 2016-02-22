"""
An example of obtaining the sample triallelic frequency spectrum for a simple two epoch demography, with selection
"""
import time
time1 = time.time()

import dadi
import numpy as np, scipy, matplotlib

nu, T = 2.0, 0.1 # instantaneous population size change (doubled in size) 0.1 time units (in 2Ne generations) ago
sig1 = 0.0 # selection coefficient for first derived allele
sig2 = 0.0 # selection coefficient for second derived allele
theta1 = 1.
theta2 = 1.
misid = 0.0 # no ancestral misidentification
dts = [0.01, 0.025, 0.001] # time steps for integration

fs = {}
for dt in dts:
    params = [nu,T,sig1,sig2,theta1,theta2,misid,dt]

    grid_pts = [40,60,80] # evaluate over these grid points, then extrapolate to $\Delta = 0$
    ns = 12 # number of observed samples

    fs0 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[0], folded = False, misid = False)
    fs1 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[1], folded = False, misid = False)
    fs2 = dadi.Triallele.demographics.two_epoch(params,ns,grid_pts[2], folded = False, misid = False)

    fs[dt] = dadi.Numerics.quadratic_extrap((fs0,fs1,fs2),(fs0.extrap_x,fs1.extrap_x,fs2.extrap_x))

tri_fs = dadi.Numerics.quadratic_extrap((fs[dts[0]],fs[dts[1]],fs[dts[2]]),(dts[0],dts[1],dts[2]))

tri_fs.to_file('tri.fs')
time2 = time.time()
print "total runtime = " + str(time2-time1)
