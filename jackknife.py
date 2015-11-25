import numpy as np

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
