import numpy as np
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
def calcB(u, n):
    B = np.zeros(n-1)
    B[0] = u*n
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
    return S

# Compute the selection linear system matrix for n+2 order terms (h != 0.5)
def calcS2(s, h, n):
    S = np.zeros((n-1, n+1))
    for i in range(0, n-1):
        S[i, i+1] = s*(1-2.0*h)*(i+2)/(n+1.0)/(n+2.0)*(i+1)*(n-i)
        S[i, i+2] = -s*(1-2.0*h)*(i+2)/(n+1.0)/(n+2.0)*(n-i-1)*(i+3)
    return S


#--------------------------------
# SFS initialisation: stady state
#--------------------------------
# Steady state (for default initialisation)
def steady_state(N, D, B, S1, S2):
    return -np.dot(np.linalg.inv(1.0/(4*N)*D+S1+S2), B)
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
def integrate_N_cst(sfs0, N, n, tf, dt, gamma=0.0, theta=1.0, h=0.5):
    # parameters of the equation
    s = gamma/N
    Tmax = tf*2.0*N
    dt = dt*2.0*N
    u = theta/(4.0*N)
    # we compute the matrices we will need
    B = calcB(u, n)
    D = calcD(n)
    J = jk.calcJK13(n) # using the order 3 jackknife approximation
    J2 = np.dot(jk.calcJK13(n+1),J)
    S1 = np.dot(calcS1(s, h , n), J)
    S2 = np.dot(calcS2(s, h , n), J2)
    Q = np.eye(n-1)-dt*(1.0/(4*N)*D+S1+S2)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        # Backward Euler scheme
        sfs = np.linalg.solve(Q,sfs+dt*B)
        t += dt
    return sfs

# for a "lambda" definition of N
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
def integrate_N_lambda(sfs0, fctN, n, tf, dt, gamma=0.0, theta=1.0, h=0.5):
    # parameters of the equation
    N0 = fctN(0)
    s = gamma/N0
    Tmax = tf*2*N0
    dt = dt*2*N0
    u = theta/(4.0*N0)
    # we compute the matrices we will need
    D = calcD(n)
    J = jk.calcJK13(n) # using the order 3 jackknife approximation
    J2 = np.dot(jk.calcJK13(n+1),J)
    S1 = np.dot(calcS1(s, h , n), J)
    S2 = np.dot(calcS2(s, h , n), J2)
    B = calcB(u, n)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        Q = np.eye(n-1)-dt*(1.0/(4*fctN(t))*D+S1+S2)
        # Backward Euler scheme
        sfs = np.linalg.solve(Q,sfs+dt*B)
        t += dt
    return sfs

# for a "lambda" definition of N - with Crank Nicholson integration scheme
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
def integrate_N_lambda_CN(sfs0, fctN, n, tf, dt, gamma=0.0, theta=1.0, h=0.5):
    # parameters of the equation
    N0 = fctN(0)
    s = gamma/N0
    Tmax = tf*2.0*N0
    dt = dt*2.0*N0
    u = theta/(4.0*N0)
    # we compute the matrices we will need
    D = calcD(n)
    J = jk.calcJK13(n) # using the order 3 jackknife approximation
    J2 = np.dot(jk.calcJK13(n+1),J)
    S1 = np.dot(calcS1(s, h , n), J)
    S2 = np.dot(calcS2(s, h , n), J2)
    B = calcB(u, n)
    # time loop:
    sfs = sfs0
    t = 0.0
    while t < Tmax:
        Q1 = np.eye(n-1)-dt/2*(1.0/(4*fctN(t))*D+S1+S2)
        Q2 = np.eye(n-1)+dt/2*(1.0/(4*fctN(t))*D+S1+S2)
        # Crank Nicholson
        sfs = np.linalg.solve(Q1,np.dot(Q2,sfs)+dt*B)
        t += dt
    return sfs

# for a "lambda" definition of N - with Crank Nicholson integration scheme
# and the iterative order 2 discrete Jackknife
# needs to start from a "good initial value" (we choose the steady state solution)
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
def integrate_CN_itJK(fctN, n, tf, dt, gamma=0.0, theta=1.0, h=0.5):
    # parameters of the equation
    N0 = fctN(0)
    s = gamma/N0
    Tmax = tf*2.0*N0
    dt = dt*2.0*N0
    u = theta/(4.0*N0)
    # we compute the matrices we will need
    D = calcD(n)
    B = calcB(u, n)
    S1 = calcS1(s, h , n)
    S2 = calcS2(s, h , n)
    # initial basis vectors for the JK
    v11 = np.ones(n)
    v21 = np.arange(1,n+1)
    v12 = np.ones(n+1)
    v22 = np.arange(1,n+2)
    # initial JK matrices
    JD1 = jk.calcJKD(n,v11,v21)
    JD2 = jk.calcJKD(n+1,v12,v22)
    S1bis = np.dot(S1, JD1)
    S2bis = np.dot(S2, np.dot(JD2, JD1))
    S = S1bis+S2bis
    sfs = initialize(N0, D, B, S1bis, S2bis)
    # time loop:
    t = 0.0
    # counter for the JK update (updates every 3 time steps)
    freq_ud = 3
    cptr = freq_ud
    while t < Tmax:
        if (cptr == freq_ud):
            # Jk updates
            for j in range(1):
                v21 = np.dot(JD1, sfs)
                JD1 = jk.calcJKD(n,v11,v21)
                v22 = np.dot(JD2, v21)
                JD2 = jk.calcJKD(n+1,v12,v22)
            # we also update the selection matrices
            S1bis = np.dot(S1, JD1)
            S2bis = np.dot(S2, np.dot(JD2, JD1))
            cptr = 1
        else: cptr += 1
        
        Q1 = np.eye(n-1)-dt/2.0*(1.0/(4*fctN(t))*D+S1bis+S2bis)
        Q2 = np.eye(n-1)+dt/2.0*(1.0/(4*fctN(t))*D+S1bis+S2bis)
        # Crank Nicholson
        sfs = np.linalg.solve(Q1,np.dot(Q2,sfs)+dt*B)
        t += dt
    return sfs