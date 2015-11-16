import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import random
import math
#-----------------------------------
# 1 dimension case
# drift and mutations - no selection
# constant N (total pop size)
#-----------------------------------

#-------------
# Parameters :
#-------------
u=0.000001
N=10000
n=1000
Tmax=5000000
dt=50
#-------------

#------------
# Functions :
#------------

# Compute the matrix A
def calcA(n):
    A=np.zeros((n-1,n-1))
    for i in range(0,n-1):
        A[i,i]=-2*(i+1)*(n-(i+1))
        if (i<n-2):
            A[i,i+1]=(n-(i+1)-1)*(i+2)
        if (i>0):
            A[i,i-1]=(n-(i+1)+1)*i
    return A
# Compute the vector B
def calcB(n):
    B=np.zeros(n-1)
    for i in range(0,n-1):
        '''if (i>30): # avoid numerical difficulties
            B[i]=0.0
        else:
            B[i]=u*misc.comb(n,i)*1/(2*N)**(i-1)*(1-1/(2*N))**(n-i)'''
        B[i]=u*10**(math.log(misc.comb(n,i),10)-(i-1)*math.log(2.0*N,10)+(n-i)*math.log(1-1/(2*N),10))
    
    return B

# Compute the linear system matrix M=(I-dt/(4N)*A)**(-1)
def calcM(A):
    Q=np.eye(n-1)-dt/(4*N)*A
    M=np.linalg.inv(Q)
    return M

# Steady state
def solstat(A,B):
    return -4*N*np.dot(np.linalg.inv(A),B)
#------------


# Initialisation
v=np.random.rand(n-1)
A=calcA(n)
B=calcB(n)
M=calcM(A)
print("Steady state : ",solstat(A,B))
#plt.plot(range(1,n),v)
#plt.show()

# Time loop to solve the system dV/dt=B+1/(4N)*AV
t=0.0
while t<Tmax:
    # Implicit Euler scheme
    v=np.dot(M,(v+dt*B))
    t=t+dt

print(v)
plt.plot(range(1,n),v)
plt.plot(range(1,n),v)
plt.show()




