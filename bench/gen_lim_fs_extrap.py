# -*- coding: UTF-8 -*-
import numpy as np
import time
import os.path

import moments
import dadi

from demographic_models_dadi import model_ooa_3D_extrap, model_YRI_CEU_extrap

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

# population expansion for dadi models
f = lambda x: 1+0.01*x
n = 80
#------------------------------
# 1D neutral N varying , T = 1:
#------------------------------
if not os.path.exists('limits/lim_1dnf_t1_'+str(n)):
    print('computing limit for 1D neutral N varying, T = 1')
    # parameters :
    T= 1.0
    h = 0.5
    g = 0
    m = 0
    # dadi limit : 
    sfsd = model_extrap1((f, T), (n, ), (g, h), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_1dnf_t1_'+str(n))


#------------------------------
# 1D neutral N varying , T = 5:
#------------------------------
if not os.path.exists('limits/lim_1dnf_t5_'+str(n)):
    print('computing limit for 1D neutral N varying, T = 5')
    # parameters :
    T= 5.0
    h = 0.5
    g = 0
    m = 0
    # dadi limit : 
    sfsd = model_extrap1((f, T), (n, ), (g, h), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_1dnf_t5_'+str(n))

#----------------------------------
# 1D, selection, N varying , T = 1:
#----------------------------------
if not os.path.exists('limits/lim_1dsf_t1_'+str(n)):
    print('computing limit for 1D, selection, N varying, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 0
    # dadi limit : 
    sfsd = model_extrap1((f, T), (n, ), (g, h), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_1dsf_t1_'+str(n))


#----------------------------------
# 1D, selection, N varying , T = 5:
#----------------------------------
if not os.path.exists('limits/lim_1dsf_t5_'+str(n)):
    print('computing limit for 1D, selection, N varying, T = 5')
    # parameters :
    T= 5.0
    h = 0.7
    g = 1.0
    m = 0
    # dadi limit : 
    sfsd = model_extrap1((f, T), (n, ), (g, h), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_1dsf_t5_'+str(n))


#------------------------------------
# 2D, selection, no migration, T = 1:
#------------------------------------
if not os.path.exists('limits/lim_2dsf_t1_'+str(n)):
    print('computing limit for 2D, selection, no migration, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 0
    # dadi limit : 
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_2dsf_t1_'+str(n))

#------------------------------------
# 2D, selection, no migration, T = 5:
#------------------------------------
if not os.path.exists('limits/lim_2dsf_t5_'+str(n)):
    print('computing limit for 2D, selection, no migration, T = 5')
    # parameters :
    T= 5.0
    h = 0.7
    g = 1.0
    m = 0
    # dadi limit : 
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_2dsf_t5_'+str(n))

#------------------------------------
# 2D, selection and migration, T = 1:
#------------------------------------
if not os.path.exists('limits/lim_2dsmf_t1_'+str(n)):
    print('computing limit for 2D, selection and migration, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 2.0
    # dadi limit : 
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_2dsmf_t1_'+str(n))

#------------------------------------
# 2D, selection and migration, T = 1:
#------------------------------------
if not os.path.exists('limits/lim_2dsmf_t5_'+str(n)):
    print('computing limit for 2D, selection and migration, T = 5')
    # parameters :
    T= 5.0
    h = 0.7
    g = 1.0
    m = 2.0
    # dadi limit : 
    sfsd = model_extrap2((f, f, T), (n, n), (g, h, m), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_2dsmf_t5_'+str(n))

#------------------------------------------------------
# 2D selection and migration croissance rapide, T = 1 :
#------------------------------------------------------
if not os.path.exists('limits/lim_2dfg_t1_'+str(n)):
    print('Selection, migration, croissance rapide 2D, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 2.0
    fexp = lambda x: np.exp(np.log(10.0)*x)
    # dadi limit : 
    sfsd = model_extrap2((fexp, fexp, T), (n, n), (g, h, m), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_2dfg_t1_'+str(n))

#---------
# YRI CEU:
#---------
if not os.path.exists('limits/lim_2d_yri_ceu_'+str(n)):
    print('computing limit for 2D, YRI-CEU')
    # parameters :
    params = [1.881, 0.0710, 1.845, 0.911, 0.355, 0.111]
    # dadi limit : 
    sfsd = model_YRI_CEU_extrap(params, (n, n), (30*n, 35*n, 40*n))
    # export : 
    sfsd.to_file('limits/lim_2d_yri_ceu_'+str(n))

#------------------------------------
# 3D selection, no migration, T = 1 :
#------------------------------------
if not os.path.exists('limits/lim_3dsf_t1_'+str(n)):
    print('processing 3D selection, no migration, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 0
    # dadi limit : 
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (5*n, 6*n, 7*n))
    # export : 
    sfsd.to_file('limits/lim_3dsf_t1_'+str(n))

#------------------------------------
# 3D selection, no migration, T = 5 :
#------------------------------------
if not os.path.exists('limits/lim_3dsf_t5_'+str(n)):
    print('processing 3D selection, no migration, T = 5')
    # parameters :
    T= 5.0
    h = 0.7
    g = 1.0
    m = 0
    # dadi limit : 
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (5*n, 6*n, 7*n))
    # export : 
    sfsd.to_file('limits/lim_3dsf_t5_'+str(n))

#------------------------------------
# 3D selection and migration, T = 1 :
#------------------------------------
if not os.path.exists('limits/lim_3dsmf_t1_'+str(n)):
    print('processing 3D selection and migration, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 2.0
    # dadi limit : 
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (5*n, 6*n, 7*n))
    # export : 
    sfsd.to_file('limits/lim_3dsmf_t1_'+str(n))
'''
#------------------------------------
# 3D selection and migration, T = 5 :
#------------------------------------
if not os.path.exists('limits/lim_3dsmf_t5_'+str(n)):
    print('processing 3D selection and migration, T = 5')
    # parameters :
    T= 5.0
    h = 0.7
    g = 1.0
    m = 2.0
    # dadi limit : 
    sfsd = model_extrap3((f, f, f, T), (n, n, n), (g, h, m), (5*n, 6*n, 7*n))
    # export : 
    sfsd.to_file('limits/lim_3dsmf_t5_'+str(n))
'''
#------------------------------------------------------
# 3D selection and migration croissance rapide, T = 1 :
#------------------------------------------------------
if not os.path.exists('limits/lim_3dfg_t1_'+str(n)):
    print('Selection, migration, croissance rapide 3D, T = 1')
    # parameters :
    T= 1.0
    h = 0.7
    g = 1.0
    m = 2.0
    fexp = lambda x: np.exp(np.log(10.0)*x)
    # dadi limit : 
    sfsd = model_extrap3((fexp, fexp, fexp, T), (n, n, n), (g, h, m), (5*n, 6*n, 7*n))
    # export : 
    sfsd.to_file('limits/lim_3dfg_t1_'+str(n))

#-------------------
# Out of africa 3D :
#-------------------
if not os.path.exists('limits/lim_ooa_3d_'+str(n)):
    print('processing out of Africa 3D')
    # parameters :
    params = [6.87846000e-01, 7.52004000e-02, 9.54548000e-02, 9.29661000e-01,
          3.55988000e-02, 2.01524000e+00, 1.49964000e+01, 7.64217000e-01,
          3.76222364e-01, 3.02770000e+00, 1.35484000e-03, 7.71636000e-01,
          2.42014000e-02]
    # dadi limit : 
    sfsd = model_ooa_3D_extrap(params, (n, n, n), (5*n, 6*n, 7*n))
    # export : 
    sfsd.to_file('limits/lim_ooa_3d_'+str(n))