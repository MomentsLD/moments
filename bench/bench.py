# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import time

import moments
import dadi

import report


#---------------------------------------------
#-----------------
# some functions :
#-----------------
def neutral_spectrum(n, ndim):
    if ndim == 1: return moments.Spectrum(np.array([0]+[1.0/i for i in range(1, n)]+[0]))
    elif ndim == 2: 
        res = np.zeros([n+1, n+1])
        res[1:-1, 0] = [1.0/i for i in range(1, n)]
        res[0, 1:-1] = [1.0/i for i in range(1, n)]
    elif ndim == 3: 
        res = np.zeros([n+1, n+1, n+1])
        res[1:-1, 0, 0] = [1.0/i for i in range(1, n)]
        res[0, 1:-1, 0] = [1.0/i for i in range(1, n)]
        res[0, 0, 1:-1] = [1.0/i for i in range(1, n)]
    return moments.Spectrum(res)

def BS_entropy(fs1, fs2, n_it, n_loop):
    tab1 = fs1.copy()
    tab2 = fs2.copy()
    # we don't take into account the values at the corners
    indfirst = tuple(np.zeros(len(tab1.shape)))
    indlast = tuple(int(n)*np.ones(len(tab1.shape)))
    tab1[indfirst] = 0.000001
    tab2[indfirst] = 0.000001
    tab1[indlast] = 0.000001
    tab2[indlast] = 0.000001
    
    res = []
    for j in range(n_loop):
        rd_ind = [tuple(np.random.choice(n, len(tab1.shape))) for i in range(n_it)]
        v1 = [tab1[x] for x in rd_ind]
        v2 = [tab2[x] for x in rd_ind]
        res.append(stats.entropy(v1, v2))
    res = np.array(res)
    # we remove the entries with infinite entropy
    pinf = float('+inf')
    res = res[res<pinf]
    return np.mean(res)

def distance(fs1, fs2, nomig = False):
    eps = 1e-16
    if not nomig: return abs(fs1-fs2)/(fs1+eps)
    else: 
        res = []
        for i in range(len(fs1)):
            if fs1[i]==0.0 or fs2[i]==0.0: res.append(0.0)
            else: res.append(abs(fs1[i]-fs2[i])/(fs1[i]))
        return res

def count_neg(sfs):
    sfs2 = sfs.copy()
    sfs2.unmask_all()
    return (sfs2<0).sum()

#--------------
# dadi models :
#--------------
def model1((nu, t), (n1, ), (g, h), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx)
    phi = dadi.Integration.one_pop(phi, xx, t, nu=nu, gamma=g, h=h)
    sfs = dadi.Spectrum.from_phi(phi, (n1, ), (xx, ))
    return sfs 

def model_extrap1((nu, t), (n1, ), (g, h), (pt1, pt2, pt3)):
    model_extrap = dadi.Numerics.make_extrap_log_func(model1)
    sfs = model_extrap((nu, t), (n1, ), (g, h), [pt1, pt2, pt3])
    return sfs

def model2((nu1, nu2, t), (n1,n2), (g, h, m), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi, xx, t, nu1, nu2, m12=m, m21=m, gamma1=g, gamma2=g, h1=h, h2=h)
    sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,xx))
    return sfs

def model_extrap2((nu1, nu2, t), (n1,n2), (g, h, m), (pt1, pt2, pt3)):
    model_extrap = dadi.Numerics.make_extrap_log_func(model2)
    sfs = model_extrap((nu1, nu2, t), (n1,n2), (g, h, m), [pt1, pt2, pt3])
    return sfs

def model3((nu1, nu2, nu3, t), (n1,n2,n3), (g, h, m), pts):
    xx = dadi.Numerics.default_grid(pts)
    phi = 0.0*dadi.PhiManip.phi_1D(xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
    phi = dadi.Integration.three_pops(phi, xx, t, nu1, nu2, nu3, m12=m, m13=m, m21=m, m23=m, m31=m, m32=m, gamma1=g, gamma2=g, gamma3=g, h1=h, h2=h, h3=h)
    sfs = dadi.Spectrum.from_phi(phi, (n,n,n), (xx,xx,xx))
    return sfs

def model_extrap3((nu1, nu2, nu3, t), (n1,n2,n3), (g, h, m), (pt1, pt2, pt3)):
    model_extrap = dadi.Numerics.make_extrap_log_func(model3)
    sfs = model_extrap((nu1, nu2, nu3, t), (n1,n2,n3), (g, h, m), [pt1, pt2, pt3])
    return sfs

#---------------------------------------------
# population expansion for dadi models
f = lambda x: 1+0.01*x
n = 30 # sample size
nb_e = 100 # number ofdrawing for the entropy computation
# we store the result to edit a report at the end...
results = []
names = []


#-------------------------
# 1D neutral equilibrium :
#-------------------------
name = 'Neutral equilibrium 1D'
print('processing '+name)

# parameters :
ndim = 1
T= 5.0
h = 0.5
g = 0
m = 0
N = 1

# analytical solution : 
neutral_fs = neutral_spectrum(n, ndim)
ref_ll = moments.Inference.ll_multinom(neutral_fs, neutral_fs)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros(n+1))
sfsm.integrate([N], [n], T)
tps_mom = time.time() - start_time
distm = distance(sfsm[1:-1], neutral_fs[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, neutral_fs, n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, neutral_fs)
print('moments: ', tps_mom, bsem, maxdm, ll, count_neg(sfsm))
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_1dn_'+str(n)):
    grid_fac = 1.5
    start_time = time.time()
    sfsd = model1((N, T), (n, ), (g, h), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_1dn_'+str(n))
    file = open('dadi_simu/time_dadi_1dn_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_1dn_'+str(n))
    file = open('dadi_simu/time_dadi_1dn_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], neutral_fs[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, neutral_fs, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, neutral_fs)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll, count_neg(sfsd))
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_1dn_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap1((N, T), (n, ), (g, h), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_1dn_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dn_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_1dn_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dn_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], neutral_fs[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, neutral_fs, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, neutral_fs)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll, count_neg(sfsd))
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])


#-------------------------------
# 1D neutral N varying T = 1.0 :
#-------------------------------
name = 'Neutral 1D, T = 1.0'
print('processing '+name)

# parameters :
ndim = 1
T= 1.0
N = lambda x: [1+0.01*x]


# we load dadi's limit : 
lim_1dnf = moments.Spectrum.from_file('limits/lim_1dnf_t1_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_1dnf, lim_1dnf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros(n+1))
sfsm.integrate(N, [n], T)
tps_mom = time.time() - start_time
distm = distance(sfsm[1:-1], lim_1dnf[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_1dnf, n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_1dnf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_1dnf_t1_'+str(n)):
    grid_fac = 1.5
    start_time = time.time()
    sfsd = model1((f, T), (n, ), (g, h), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_1dnf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_1dnf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_1dnf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_1dnf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dnf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dnf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dnf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_1dnf_t1_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap1((f, T), (n, ), (g, h), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_1dnf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dnf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_1dnf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dnf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dnf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dnf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dnf)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#-------------------------------
# 1D neutral N varying T = 5.0 :
#-------------------------------
name = 'Neutral 1D, T = 5.0'
print('processing '+name)

# parameters :
ndim = 1
T= 5.0
N = lambda x: [1+0.01*x]


# we load dadi's limit : 
lim_1dnf = moments.Spectrum.from_file('limits/lim_1dnf_t5_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_1dnf, lim_1dnf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros(n+1))
sfsm.integrate(N, [n], T)
tps_mom = time.time() - start_time
distm = distance(sfsm[1:-1], lim_1dnf[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_1dnf, n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_1dnf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_1dnf_t5_'+str(n)):
    grid_fac = 1.5
    start_time = time.time()
    sfsd = model1((f, T), (n, ), (g, h), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_1dnf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_1dnf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_1dnf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_1dnf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dnf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dnf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dnf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_1dnf_t5_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap1((f, T), (n, ), (g, h), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_1dnf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dnf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_1dnf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dnf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dnf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dnf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dnf)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#-----------------------------------
# 1D, selection, N varying T = 1.0 :
#-----------------------------------
name = 'Selection 1D, T = 1.0'
print('processing '+name)

# parameters :
ndim = 1
T= 1.0
h = 0.7
g = 1.0
N = lambda x: [1+0.01*x]


# we load dadi's limit : 
lim_1dsf = moments.Spectrum.from_file('limits/lim_1dsf_t1_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_1dsf, lim_1dsf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros(n+1))
sfsm.integrate(N, [n], T, gamma=g, h=h)
tps_mom = time.time() - start_time
distm = distance(sfsm[1:-1], lim_1dsf[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_1dsf, n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_1dsf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 2*n : 
if not os.path.exists('dadi_simu/dadi_1dsf_t1_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model1((f, T), (n, ), (g, h), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_1dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_1dsf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_1dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_1dsf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dsf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dsf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dsf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_1dsf_t1_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap1((f, T), (n, ), (g, h), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_1dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dsf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_1dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dsf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dsf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dsf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dsf)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#-----------------------------------
# 1D, selection, N varying T = 5.0 :
#-----------------------------------
name = 'Selection 1D, T = 5.0'
print('processing '+name)

# parameters :
ndim = 1
T= 5.0
h = 0.7
g = 1.0
N = lambda x: [1+0.01*x]


# we load dadi's limit : 
lim_1dsf = moments.Spectrum.from_file('limits/lim_1dsf_t5_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_1dsf, lim_1dsf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros(n+1))
sfsm.integrate(N, [n], T, gamma=g, h=h)
tps_mom = time.time() - start_time
distm = distance(sfsm[1:-1], lim_1dsf[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_1dsf, n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_1dsf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 2*n : 
if not os.path.exists('dadi_simu/dadi_1dsf_t5_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model1((f, T), (n, ), (g, h), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_1dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_1dsf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_1dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_1dsf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dsf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dsf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dsf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_1dsf_t5_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap1((f, T), (n, ), (g, h), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_1dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dsf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_1dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_1dsf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[1:-1], lim_1dsf[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_1dsf, n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_1dsf)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#-----------------------------------------------------------------------------------
#-------------------------
# 2D neutral equilibrium :
#-------------------------
name = 'Neutral equilibrium 2D'
print('processing '+name)

# parameters :
ndim = 2
T= 5.0
h = 0.5
g = 0
m = 0
N = 1

# analytical solution : 
neutral_fs = neutral_spectrum(n, ndim)
ref_ll = moments.Inference.ll_multinom(neutral_fs, neutral_fs)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1]))
sfsm.integrate([N, N], [n, n], T)
tps_mom = time.time() - start_time
distm = distance(sfsm[0,1:-1], neutral_fs[0,1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm[0,:], neutral_fs[0,:], n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, neutral_fs)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_2dn_'+str(n)):
    grid_fac = 1.5
    start_time = time.time()
    sfsd = model2((N, N, T), (n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_2dn_'+str(n))
    file = open('dadi_simu/time_dadi_2dn_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_2dn_'+str(n))
    file = open('dadi_simu/time_dadi_2dn_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[0,1:-1], neutral_fs[0,1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd[0,:], neutral_fs[0,:], n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, neutral_fs)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_2dn_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap2((N, N, T), (n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_2dn_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dn_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_2dn_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dn_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[0,1:-1], neutral_fs[0,1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd[0,:], neutral_fs[0,:], n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, neutral_fs)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 2D selection, no migration, T = 1 :
#------------------------------------
name = 'Selection 2D, T = 1.0'
print('processing '+name)

# parameters :
ndim = 2
T= 1.0
h = 0.7
g = 1.0
m = 0
N = lambda x: [1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_2dsf = moments.Spectrum.from_file('limits/lim_2dsf_t1_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_2dsf, lim_2dsf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1]))
sfsm.integrate(N, [n, n], T, gamma=gamma, h=hh)
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**2)[1:-1], lim_2dsf.reshape((n+1)**2)[1:-1], True)
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_2dsf, n**2, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_2dsf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_2dsf_t1_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model2((f, f, T), (n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_2dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_2dsf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_2dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_2dsf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**2)[1:-1], lim_2dsf.reshape((n+1)**2)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_2dsf, n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_2dsf_t1_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    distd = distance(sfsd.reshape((n+1)**2)[1:-1], lim_2dsf.reshape((n+1)**2)[1:-1])
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_2dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_2dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_2dsf, n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 2D selection, no migration, T = 5 :
#------------------------------------
name = 'Selection 2D, T = 5.0'
print('processing '+name)

# parameters :
ndim = 2
T= 5.0
h = 0.7
g = 1.0
m = 0
N = lambda x: [1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_2dsf = moments.Spectrum.from_file('limits/lim_2dsf_t5_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_2dsf, lim_2dsf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1]))
sfsm.integrate(N, [n, n], T, gamma=gamma, h=hh)
tps_mom = time.time() - start_time
distm = distance(sfsm[0,:], lim_2dsf[0,:])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm[0,:], lim_2dsf[0,:], n**2, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_2dsf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]
print(sfsm[0,1:4])

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_2dsf_t5_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model2((f, f, T), (n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_2dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_2dsf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_2dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_2dsf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[0,:], lim_2dsf[0,:])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd[0,:], lim_2dsf[0,:], n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_2dsf_t5_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_2dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_2dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[0,:], lim_2dsf[0,:])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd[0,:], lim_2dsf[0,:], n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsf)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 2D selection and migration, T = 1 :
#------------------------------------
name = 'Selection, migration 2D, T = 1.0'
print('processing '+name)

# parameters :
ndim = 2
T= 1.0
h = 0.7
g = 1.0
m = 2.0
N = lambda x: [1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_2dsmf = moments.Spectrum.from_file('limits/lim_2dsmf_t1_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_2dsmf, lim_2dsmf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1]))
sfsm.integrate(N, [n, n], T, gamma=gamma, h=hh, m=m*np.ones([ndim, ndim]))
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**2)[1:-1], lim_2dsmf.reshape((n+1)**2)[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_2dsmf, n**2, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_2dsmf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_2dsmf_t1_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model2((f, f, T), (n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_2dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_2dsmf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_2dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_2dsmf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**2)[1:-1], lim_2dsmf.reshape((n+1)**2)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_2dsmf, n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsmf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_2dsmf_t1_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_2dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsmf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_2dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsmf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**2)[1:-1], lim_2dsmf.reshape((n+1)**2)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_2dsmf, n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsmf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 2D selection and migration, T = 5 :
#------------------------------------
name = 'Selection, migration 2D, T = 5.0'
print('processing '+name)

# parameters :
ndim = 2
T= 5.0
h = 0.7
g = 1.0
m = 2.0
N = lambda x: [1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_2dsmf = moments.Spectrum.from_file('limits/lim_2dsmf_t5_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_2dsmf, lim_2dsmf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1]))
sfsm.integrate(N, [n, n], T, gamma=gamma, h=hh, m=m*np.ones([ndim, ndim]))
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**2)[1:-1], lim_2dsmf.reshape((n+1)**2)[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_2dsmf, n**2, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_2dsmf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 2*n : 
if not os.path.exists('dadi_simu/dadi_2dsmf_t5_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model2((f, f, T), (n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_2dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_2dsmf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_2dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_2dsmf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**2)[1:-1], lim_2dsmf.reshape((n+1)**2)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_2dsmf, n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsmf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_2dsmf_t5_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_2dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsmf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_2dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_2dsmf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**2)[1:-1], lim_2dsmf.reshape((n+1)**2)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_2dsmf, n**2, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_2dsmf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#-----------------------------------------------------------------------------------
#-------------------------
# 3D neutral equilibrium :
#-------------------------
name = 'Neutral equilibrium 3D'
print('processing '+name)

# parameters :
ndim = 3
T= 5.0
h = 0.5
g = 0
m = 0
N = 1

# analytical solution : 
neutral_fs = neutral_spectrum(n, ndim)
ref_ll = moments.Inference.ll_multinom(neutral_fs, neutral_fs)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1, n+1]))
sfsm.integrate([N, N, N], [n, n, n], T)
tps_mom = time.time() - start_time
distm = distance(sfsm[0, 1:-1, 0], neutral_fs[0, 1:-1, 0])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm[0, :, 0], neutral_fs[0, :, 0], n, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, neutral_fs)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_3dn_'+str(n)):
    grid_fac = 1.5
    start_time = time.time()
    sfsd = model3((N, N, N, T), (n, n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_3dn_'+str(n))
    file = open('dadi_simu/time_dadi_3dn_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_3dn_'+str(n))
    file = open('dadi_simu/time_dadi_3dn_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[0, 1:-1, 0], neutral_fs[0, 1:-1, 0])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd[0, :, 0], neutral_fs[0, :, 0], n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, neutral_fs)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_3dn_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap3((N, N, N, T), (n, n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_3dn_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dn_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_3dn_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dn_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd[0, 1:-1, 0], neutral_fs[0, 1:-1, 0])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd[0, :, 0], neutral_fs[0, :, 0], n, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, neutral_fs)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 3D selection, no migration, T = 1 :
#------------------------------------
name = 'Selection 3D, T = 1.0'
print('processing '+name)

# parameters :
ndim = 3
T= 1.0
h = 0.7
g = 1.0
m = 0
N = lambda x: [1+0.01*x, 1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_3dsf = moments.Spectrum.from_file('limits/lim_3dsf_t1_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_3dsf, lim_3dsf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1, n+1]))
sfsm.integrate(N, [n, n, n], T, gamma=gamma, h=hh)
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**3)[1:-1], lim_3dsf.reshape((n+1)**3)[1:-1], True)
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_3dsf, n**3, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_3dsf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_3dsf_t1_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model3((f, f, f, T), (n, n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_3dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_3dsf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_3dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_3dsf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_3dsf_t1_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_3dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_3dsf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 3D selection, no migration, T = 5 :
#------------------------------------
name = 'Selection 3D, T = 5.0'
print('processing '+name)

# parameters :
ndim = 3
T= 5.0
h = 0.7
g = 1.0
m = 0
N = lambda x: [1+0.01*x, 1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_3dsf = moments.Spectrum.from_file('limits/lim_3dsf_t5_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_3dsf, lim_3dsf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1, n+1]))
sfsm.integrate(N, [n, n, n], T, gamma=gamma, h=hh)
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**3)[1:-1], lim_3dsf.reshape((n+1)**3)[1:-1], True)
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_3dsf, n**3, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_3dsf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 1.5*n : 
if not os.path.exists('dadi_simu/dadi_3dsf_t5_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model3((f, f, f, T), (n, n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_3dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_3dsf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_3dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_3dsf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_3dsf_t5_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_3dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_3dsf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 3D selection and migration, T = 1 :
#------------------------------------
name = 'Selection, migration 3D, T = 1.0'
print('processing '+name)

# parameters :
ndim = 3
T= 1.0
h = 0.7
g = 1.0
m = 2.0
N = lambda x: [1+0.01*x, 1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_3dsmf = moments.Spectrum.from_file('limits/lim_3dsmf_t1_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_3dsmf, lim_3dsmf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1, n+1]))
sfsm.integrate(N, [n, n, n], T, gamma=gamma, h=hh, m=m*np.ones([ndim, ndim]))
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**3)[1:-1], lim_3dsmf.reshape((n+1)**3)[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_3dsmf, n**3, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_3dsmf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 2*n : 
if not os.path.exists('dadi_simu/dadi_3dsmf_t1_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model3((f, f, f, T), (n, n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_3dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_3dsmf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_3dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_3dsmf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsmf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsmf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsmf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_3dsmf_t1_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_3dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsmf_t1_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_3dsmf_t1_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsmf_t1_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsmf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsmf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsmf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#------------------------------------
# 3D selection and migration, T = 5 :
#------------------------------------
name = 'Selection, migration 3D, T = 5.0'
print('processing '+name)

# parameters :
ndim = 3
T= 5.0
h = 0.7
g = 1.0
m = 2.0
N = lambda x: [1+0.01*x, 1+0.01*x, 1+0.01*x]
gamma = g*np.ones(ndim)
hh = h*np.ones(ndim)

# we load dadi's limit : 
lim_3dsmf = moments.Spectrum.from_file('limits/lim_3dsmf_t5_'+str(n))
ref_ll = moments.Inference.ll_multinom(lim_3dsmf, lim_3dsmf)

# moments : 
start_time = time.time()
sfsm = moments.Spectrum(np.zeros([n+1, n+1, n+1]))
sfsm.integrate(N, [n, n, n], T, gamma=gamma, h=hh, m=m*np.ones([ndim, ndim]))
tps_mom = time.time() - start_time
distm = distance(sfsm.reshape((n+1)**3)[1:-1], lim_3dsmf.reshape((n+1)**3)[1:-1])
maxdm = np.amax(distm)
bsem = BS_entropy(sfsm, lim_3dsmf, n**3, nb_e)
ll = ref_ll - moments.Inference.ll_multinom(sfsm, lim_3dsmf)
print('moments: ', tps_mom, bsem, maxdm, ll)
resm = [tps_mom, bsem, maxdm, ll, count_neg(sfsm)]

# dadi 1, pts = 2*n : 
if not os.path.exists('dadi_simu/dadi_3dsmf_t5_'+str(n)):
    grid_fac = 2
    start_time = time.time()
    sfsd = model3((f, f, f, T), (n, n, n), (g, h, m), grid_fac * n)
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_3dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_3dsmf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_3dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_3dsmf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsmf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsmf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsmf)
print('dadi 1: ', tps_dadi, bsed, maxdd, ll)
resd1 = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

# dadi Richardson extrapolation : 
if not os.path.exists('dadi_simu/dadi_extrap_3dsmf_t5_'+str(n)):
    start_time = time.time()
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (n, n+10, n+20))
    tps_dadi = time.time() - start_time
    # export
    sfsd.to_file('dadi_simu/dadi_extrap_3dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsmf_t5_'+str(n), "w")
    file.write(str(tps_dadi))
    file.close()
else:
    sfsd = dadi.Spectrum.from_file('dadi_simu/dadi_extrap_3dsmf_t5_'+str(n))
    file = open('dadi_simu/time_dadi_extrap_3dsmf_t5_'+str(n), 'r')
    tps_dadi = float(file.read())
distd = distance(sfsd.reshape((n+1)**3)[1:-1], lim_3dsmf.reshape((n+1)**3)[1:-1])
maxdd = np.amax(distd)
bsed = BS_entropy(sfsd, lim_3dsmf, n**3, nb_e)
ll = ref_ll - dadi.Inference.ll_multinom(sfsd, lim_3dsmf)
sfsd2 = moments.Spectrum(sfsd)
print('dadi extrap: ', tps_dadi, bsed, maxdd, ll)
resde = [tps_dadi, bsed, maxdd, ll, count_neg(sfsd)]

names.append(name)
results.append([resd1, resde, resm])

#-----------------------------------------------------------------------------------
#report.generate_tex_table(results, names)
report.generate_formated_table(results, names)

os.system("pdflatex report.tex")
