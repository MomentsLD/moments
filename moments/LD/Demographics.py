import numpy as np
from moments.LD import Numerics
from moments.LD import Corrections

def equilibrium(order, ns=200, rho=0, theta=0.0008, corrected=False, ism=False):
    y = Numerics.equilibrium(order=order, rho=rho, theta=theta, ism=ism)
    if corrected == True:
        return Corrections.corrected_onepop(y, n=ns, order=order)
    else:
        return y

def two_epoch(order, params, ns=200, rho=0, theta=0.008, corrected=False, ism=False):
    nu,T = params
    y = equilibrium(order, rho=rho, theta=theta, ism=ism)
    y = Numerics.integrate(y, T, rho=rho, theta=theta, nu=nu, order=order, dt=0.001, ism=ism)
    if corrected == True:
        return Corrections.corrected_onepop(y, n=ns, order=order)
    else:
        return y

def three_epoch(order, params, ns=200, rho=0,  theta=0.008, corrected=False, ism=False):
    nu1,nu2,T1,T2 = params
    y = equilibrium(order, rho=rho, theta=theta, ism=ism)
    y = Numerics.integrate(y, T1, rho=rho, theta=theta, nu=nu1, order=order, dt=0.001, ism=ism)
    y = Numerics.integrate(y, T2, rho=rho, theta=theta, nu=nu2, order=order, dt=0.001, ism=ism)
    if corrected == True:
        return Corrections.corrected_onepop(y, n=ns, order=order)
    else:
        return y

def growth(order, params, ns=200, rho=0,  theta=0.008, corrected=False, ism=False):
    nuF,T = params
    y = equilibrium(order, rho=rho, theta=theta, ism=ism)
    nu_func = lambda t: np.exp( np.log(nuF) *t/T)
    y = Numerics.integrate(y, T, rho=rho, theta=theta, nu=nu_func, order=order, dt=0.001, ism=ism)
    if corrected == True:
        return Corrections.corrected_onepop(y, n=ns, order=order)
    else:
        return y

def bottlegrowth(order, params, ns=200, rho=0,  theta=0.008, corrected=False, ism=False):    
    nuB,nuF,T = params
    y = equilibrium(order, rho=rho, theta=theta, ism=ism)
    nu_func = lambda t: nuB * np.exp( np.log(nuF/nuB) *t/T)
    y = Numerics.integrate(y, T, rho=rho, theta=theta, nu=nu_func, order=order, dt=0.001, ism=ism)
    if corrected == True:
        return Corrections.corrected_onepop(y, n=ns, order=order)
    else:
        return y

