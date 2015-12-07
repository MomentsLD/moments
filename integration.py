import numpy as np
import scipy.misc as misc
import math

import jackknife as jk

#--------------------------------------------------
# Functions for the computation of the Phi-moments:
# we integrate the ode system on the Phi_n(i) to compute their evolution
# we write it (and solve it) as an approximated linear system:
#       Phi_n' = Bn(N) + (1/(4N)Dn + S1n + S2n)Phi_n
# where :
#       N is the total population size
#       Bn(N) is the mutation source term
#       1/(4N)Dn is the drift effect matrix
#       S1n is the selection matrix for h = 0.5
#       S2n is the effect of h != 0.5
#--------------------------------------------------


#------------------------------------------
# Matrices and vectors for the linear system
#-------------------------------------------
# Compute the vector B (mutation source term)
def calcB(u, n, N):
    B = np.zeros(n-1)
    for i in range(0, n-1):
        B[i] = u*10**(math.log(misc.comb(n, i+1), 10)-i*math.log(2.0*N, 10)+(n-i-1)
               *math.log(1-1/(2*N), 10))
    return B


# Compute the matrix D (drift)
def calcD(n):
    D = np.zeros((n-1, n-1))
    for i in range(0, n-1):
        D[i,i] = -2*(i+1)*(n-(i+1))
        if (i < n-2):
            D[i, i+1] = (n-(i+1)-1)*(i+2)
        if (i > 0):
            D[i, i-1] = (n-(i+1)+1)*i
    return D

# Compute the selection linear system matrix for n+1 order terms
def calcS1(s, h, n):
    S = np.zeros((n-1, n))
    for i in range(0, n-1):
        S[i, i] = s*h*(i+1)*(n-i)/(n+1)
        S[i, i+1] = -s*h/(n+1)*(n-i-1)*(i+2)
    J = jk.calcJK13(n) # using the order 3 jackknife approximation
    return np.dot(S, J)

# Compute the selection linear system matrix for n+2 order terms (h != 0.5)
def calcS2(s, h, n):
    S = np.zeros((n-1, n+1))
    for i in range(0, n-1):
        S[i, i+1] = s*(1-2*h)*(i+2)/(n+1)/(n+2)*(i+1)*(n-i)
        S[i, i+2] = -s*(1-2*h)*(i+2)/(n+1)/(n+2)*(n-i-1)*(i+3)
    J = jk.calcJK13(n) # using the order 3 jackknife approximation
    J2 = np.dot(jk.calcJK13(n+1),J)
    return np.dot(S, J2)


#--------------------------------
# SFS initialisation: stady state
#--------------------------------
# Steady state (for default initialisation)
def steady_state(N, D, B, S1, S2):
    return -np.dot(np.linalg.inv(1/(4*N)*D+S1+S2), B)
# alias
initialize = steady_state

#--------------------
# Integration in time
#--------------------
# N : total population size
# n : sample size
# tf : final simulation time (/2N generations)
# gamma : selection coefficient (same as in dadi)
# u : mutation rate (*4N)
# h : allele dominance

# for a constant N
def integrate_N_cst(sfs0, N, n, tf, dt, gamma=0, u=1, h=0.5):
    # parameters of the equation
    s = gamma/N
    Tmax = tf*2*N
    u /= 4*N
    # we compute the matrices we will need
    B = calcB(u, n, N)
    D = calcD(n)
    S1 = calcS1(s, h , n)
    S2 = calcS2(s, h , n)
    Q = np.eye(n-1)-dt*(1/(4*N)*D+S1+S2)
    M = np.linalg.inv(Q)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        # Backward Euler scheme
        sfs = np.dot(M, (sfs+dt*B))
        t += dt
    return sfs

# for a "lambda" definition of N
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
def integrate_N_lambda(sfs0, fctN, n, tf, dt, gamma=0, u=1, h=0.5):
    # parameters of the equation
    N0 = fctN(0)
    s = gamma/N0
    Tmax = tf*2*N0
    u /= 4*N0
    # we compute the matrices we will need
    D = calcD(n)
    S1 = calcS1(s, h , n)
    S2 = calcS2(s, h , n)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        B = calcB(u, n, fctN(t))
        Q = np.eye(n-1)-dt*(1/(4*fctN(t))*D+S1+S2)
        # Backward Euler scheme
        sfs = np.linalg.solve(Q,sfs+dt*B)
        #M = np.linalg.inv(Q)
        #sfs = np.dot(M, (sfs+dt*B))
        t += dt
    return sfs

# for a "lambda" definition of N - with Crank Nicholson integration scheme
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
def integrate_N_lambda_CN(sfs0, fctN, n, tf, dt, gamma=0, u=1, h=0.5):
    # parameters of the equation
    N0 = fctN(0)
    s = gamma/N0
    Tmax = tf*2*N0
    u /= 4*N0
    # we compute the matrices we will need
    D = calcD(n)
    S1 = calcS1(s, h , n)
    S2 = calcS2(s, h , n)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        B = calcB(u, n, fctN(t))
        Q1 = np.eye(n-1)-dt/2*(1/(4*fctN(t))*D+S1+S2)
        Q2 = np.eye(n-1)+dt/2*(1/(4*fctN(t))*D+S1+S2)
        # Crank Nicholson
        sfs = np.linalg.solve(Q1,np.dot(Q2,sfs)+dt*B)
        #M = np.linalg.inv(Q1)
        #sfs = np.dot(M, np.dot(Q2, sfs)+dt*B)
        t += dt
    return sfs

# for a "lambda" definition of N - with 4th order RK schemme
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
def integrate_N_lambda_RK4(sfs0, fctN, n, tf, dt, gamma=0, u=1, h=0.5):
    # parameters of the equation
    N0 = fctN(0)
    s = gamma/N0
    Tmax = tf*2*N0
    u /= 4*N0
    # we compute the matrices we will need
    D = calcD(n)
    S1 = calcS1(s, h , n)
    S2 = calcS2(s, h , n)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        B = calcB(u, n, fctN(t))
        A = 1/(4*fctN(t))*D+S1+S2
        # Runge Kutta 4
        k1 = np.dot(A,sfs)+B
        k2 = np.dot(A,sfs+dt/2*k1)+B
        k3 = np.dot(A,sfs+dt/2*k2)+B
        k4 = np.dot(A,sfs+dt*k3)+B
        sfs += dt/6*(k1+2*k2+2*k3+k4)
        t += dt
    return sfs


