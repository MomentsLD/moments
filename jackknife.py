import numpy as np
import math
#--------------------------------------------
# Jackknife extrapolations :
# used for the moment closure under selection
# to extrapolate the Phi_(n+1) and Phi_(n+2)
# from the Phi_n.
#-----------------------------------

# The choice i' in n samples that best approximates the frequency of \i/(n + 1) is i*n / (n + 1)
def index_bis(i,n):
    return int(max(round(i*n/(n+1)),2))

# Compute the order 2 Jackknife extrapolation coefficients for 1 jump (Phi_n -> Phi_(n+1))
def calcJK12(n):
    J = np.zeros((n, n-1))
    for i in range(1, n-1):
        J[i,i] = (1+n)*(2-(i+1)+n)/(2+n)/(3+n)
        J[i,i-1] = (i+2)*(n+1)/(n+2)/(n+3)
    J[0, 0] = (1+n)*(5+n)/(n+2)/(n+3)
    J[0, 1] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-3] = -2*(1+n)/(n+2)/(n+3)
    J[n-1, n-2] = (1+n)*(5+n)/(n+2)/(n+3)
    return J

# Compute the order 3 Jackknife extrapolation coefficients for 1 jump (Phi_n -> Phi_(n+1))
def calcJK13(n):
    J = np.zeros((n, n-1))
    for i in range(0, n):
        ibis = index_bis(i+1,n)-1
        if (i == n-1): ibis -= 1
        J[i,ibis] = -(1+n)*((2+i)*(2+n)*(-6-n+(i+1)*(3+n))-2*(4+n)*(-1+(i+1)*(2+n))*(ibis+1)
                  +(12+7*n+n**2)*(ibis+1)**2)/(2+n)/(3+n)/(4+n)
        J[i,ibis-1] = (1+n)*(4+(1+i)**2*(6+5*n+n**2)-(i+1)*(14+9*n+n**2)-(4+n)*(-5-n+2*(i+1)*(2+n))*(ibis+1)
                    +(12+7*n+n**2)*(ibis+1)**2)/(2+n)/(3+n)/(4+n)/2
        J[i,ibis+1] = (1+n)*((2+i)*(2+n)*(-2+(i+1)*(3+n))-(4+n)*(1+n+2*(i+1)*(2+n))*(ibis+1)
                    +(12+7*n+n**2)*(ibis+1)**2)/(2+n)/(3+n)/(4+n)/2
    return J

# Compute the order 2 Jackknife extrapolation coefficients for 2 jumps (Phi_n -> Phi_(n+2))
def calcJK22(n):
    J = np.zeros((n+1, n-1))
    for i in range(1, n-1):
        J[i,i] = (1+n)*(2-2*(i+1)+n)/(4+n)/(3+n)
        J[i,i-1] = 2*(i+2)*(n+1)/(n+4)/(n+3)
    J[0,0] = (8+9*n+n**2)/(12+7*n+n**2)
    J[0,1] = 4*(1+n)/(12+7*n+n**2)
    J[n-1,n-2] = 6*(1+n)/(12+7*n+n**2)
    J[n-1,n-3] = (-2-n+n**2)/(12+7*n+n**2)
    J[n,n-2] = (8+9*n+n**2)/(12+7*n+n**2)
    J[n,n-3] = 4*(1+n)/(12+7*n+n**2)
    return J

# Compute the order 3 Jackknife extrapolation coefficients for 2 jumps (Phi_n -> Phi_(n+2))
def calcJK23(n):
    J = np.zeros((n+1, n-1))
    for i in range(0, n+1):
        ibis = index_bis(i+1,n)-1
        if (i == n-1) or (i == n): ibis = n-3
        J[i,ibis] = -(1+n)*((2+i)*(2+n)*(-9-n+(i+1)*(3+n))-2*(5+n)*(-2+(i+1)*(2+n))*(ibis+1)
                  +(20+9*n+n**2)*(ibis+1)**2)/(3+n)/(4+n)/(5+n)
        J[i,ibis-1] = (1+n)*(12+(1+i)**2*(6+5*n+n**2)-(i+1)*(22+13*n+n**2)-(5+n)*(-8-n+2*(i+1)*(2+n))*(ibis+1)
                    +(20+9*n+n**2)*(ibis+1)**2)/(3+n)/(4+n)/(5+n)/2
        J[i,ibis+1] = (1+n)*((2+i)*(2+n)*(-4+(i+1)*(3+n))-(5+n)*(n+2*(i+1)*(2+n))*(ibis+1)
                    +(20+9*n+n**2)*(ibis+1)**2)/(3+n)/(4+n)/(5+n)/2
    return J


# Discrete iterative Jakknife
def index(i,n):
    return int(min(max(math.floor(i*n/(n+1)),1),n-2))

# Compute the matrix to transform Phi_(n+1)->Phi_n
def matA(n):
    A = np.zeros((n-1, n))
    for i in range(0, n-1):
        A[i, i] = (n-i)/(n+1)
        A[i, i+1] = (i+2)/(n+1)
    return A

# Disctrete JK using Cramer formula
def calcJKD(n,v1,v2):
    J = np.zeros((n, n-1))
    A = matA(n)
    # cas général
    for i in range(1,n-1):
        a1 = np.longdouble(v1[i]*A[i-1,i]+v1[i-1]*A[i-1,i-1])
        a2 = np.longdouble(v1[i]*A[i,i]+v1[i+1]*A[i,i+1])
        a3 = np.longdouble(v2[i]*A[i-1,i]+v2[i-1]*A[i-1,i-1])
        a4 = np.longdouble(v2[i]*A[i,i]+v2[i+1]*A[i,i+1])
        J[i,i-1] = np.longdouble((v1[i]*a4-v2[i]*a2)/(a1*a4-a3*a2))
        J[i,i] = np.longdouble((a1*v2[i]-a3*v1[i])/(a1*a4-a3*a2))
    # case i=0
    a1 = np.longdouble(v1[0]*A[0,0]+v1[1]*A[0,1])
    a2 = np.longdouble(v1[1]*A[1,1]+v1[2]*A[1,2])
    a3 = np.longdouble(v2[0]*A[0,0]+v2[1]*A[0,1])
    a4 = np.longdouble(v2[1]*A[1,1]+v2[2]*A[1,2])
    J[0,0] = np.longdouble((v1[0]*a4-v2[0]*a2)/(a1*a4-a3*a2))
    J[0,1] = np.longdouble((a1*v2[0]-a3*v1[0])/(a1*a4-a3*a2))

    # case i=n-1
    a1 = np.longdouble(v1[n-1]*A[n-2,n-1]+v1[n-2]*A[n-2,n-2])
    a2 = np.longdouble(v1[n-2]*A[n-3,n-2]+v1[n-3]*A[n-3,n-3])
    a3 = np.longdouble(v2[n-1]*A[n-2,n-1]+v2[n-2]*A[n-2,n-2])
    a4 = np.longdouble(v2[n-2]*A[n-3,n-2]+v2[n-3]*A[n-3,n-3])
    J[n-1,n-2] = np.longdouble((v1[n-1]*a4-v2[n-1]*a2)/(a1*a4-a3*a2))
    J[n-1,n-3] = np.longdouble((a1*v2[n-1]-a3*v1[n-1])/(a1*a4-a3*a2))
    return J

# with np.linajg.solve
def calcJKD2(n,v1,v2):
    J = np.zeros((n, n-1))
    A = matA(n)
    # cas général
    for i in range(1,n-1):
        m = np.array([[v1[i]*A[i-1,i]+v1[i-1]*A[i-1,i-1], v1[i]*A[i,i]+v1[i+1]*A[i,i+1]], [v2[i]*A[i-1,i]+v2[i-1]*A[i-1,i-1], v2[i]*A[i,i]+v2[i+1]*A[i,i+1]]])
        j = np.linalg.solve(m,np.array([v1[i], v2[i]]))
        J[i,i-1] = j[0]
        J[i,i] = j[1]
    # cas i=0
    m = np.array([[v1[0]*A[0,0]+v1[1]*A[0,1], v1[1]*A[1,1]+v1[2]*A[1,2]], [v2[0]*A[0,0]+v2[1]*A[0,1], v2[1]*A[1,1]+v2[2]*A[1,2]]])
    j = np.linalg.solve(m,np.array([v1[0], v2[0]]))
    J[0,0] = j[0]
    J[0,1] = j[1]

    # cas i=n-1
    m = np.array([[v1[n-1]*A[n-2,n-1]+v1[n-2]*A[n-2,n-2], v1[n-2]*A[n-3,n-2]+v1[n-3]*A[n-3,n-3]], [v2[n-1]*A[n-2,n-1]+v2[n-2]*A[n-2,n-2], v2[n-2]*A[n-3,n-2]+v2[n-3]*A[n-3,n-3]]])
    j = np.linalg.solve(m,np.array([v1[n-1], v2[n-1]]))
    J[n-1,n-2] = j[0]
    J[n-1,n-3] = j[1]
    return J

# Disctrete JK using Cramer formula
def calcJKD3(n,v1,v2):
    J = np.zeros((n, n-1))
    A = matA(n)
    # cas général
    for i in range(1,n-1):
        a1 = v1[i]*A[i-1,i]+v1[i-1]*A[i-1,i-1]
        a2 = v1[i]*A[i,i]+v1[i+1]*A[i,i+1]
        a3 = v2[i]*A[i-1,i]+v2[i-1]*A[i-1,i-1]
        a4 = v2[i]*A[i,i]+v2[i+1]*A[i,i+1]
        J[i,i-1] = (v1[i]*a4-v2[i]*a2)/(a1*a4-a3*a2)
        J[i,i] = (a1*v2[i]-a3*v1[i])/(a1*a4-a3*a2)

    # case i=0
    a1 = v1[0]*A[0,0]+v1[1]*A[0,1]
    a2 = v1[1]*A[1,1]+v1[2]*A[1,2]
    a3 = v2[0]*A[0,0]+v2[1]*A[0,1]
    a4 = v2[1]*A[1,1]+v2[2]*A[1,2]
    J[0,0] = (v1[0]*a4-v2[0]*a2)/(a1*a4-a3*a2)
    J[0,1] = (a1*v2[0]-a3*v1[0])/(a1*a4-a3*a2)

    # case i=n-1
    a1 = v1[n-1]*A[n-2,n-1]+v1[n-2]*A[n-2,n-2]
    a2 = v1[n-2]*A[n-3,n-2]+v1[n-3]*A[n-3,n-3]
    a3 = v2[n-1]*A[n-2,n-1]+v2[n-2]*A[n-2,n-2]
    a4 = v2[n-2]*A[n-3,n-2]+v2[n-3]*A[n-3,n-3]
    J[n-1,n-2] = (v1[n-1]*a4-v2[n-1]*a2)/(a1*a4-a3*a2)
    J[n-1,n-3] = (a1*v2[n-1]-a3*v1[n-1])/(a1*a4-a3*a2)
    return J



