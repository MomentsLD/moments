from moments.LD.Parsing import gcs as geno_calc
import numpy as np
import unittest
import time

def compute_D(counts):
    n1 = counts[:,0]
    n2 = counts[:,1]
    n3 = counts[:,2]
    n4 = counts[:,3]
    n5 = counts[:,4]
    n6 = counts[:,5]
    n7 = counts[:,6]
    n8 = counts[:,7]
    n9 = counts[:,8]
    n = np.sum(counts, axis=1)
    denom = np.zeros(n.shape)
    numer = np.zeros(n.shape)
    for i in range(len(n)):
        numer[i] = (-(n2[i]*n4[i])/4. - (n3[i]*n4[i])/2. + (n1[i]*n5[i])/4. - (n3[i]*n5[i])/4. + (n1[i]*n6[i])/2. + (n2[i]*n6[i])/4. - (n2[i]*n7[i])/2. - n3[i]*n7[i] - (n5[i]*n7[i])/4. - (n6[i]*n7[i])/2. + (n1[i]*n8[i])/2. - (n3[i]*n8[i])/2. + (n4[i]*n8[i])/4. - (n6[i]*n8[i])/4. + n1[i]*n9[i] + (n2[i]*n9[i])/2. + (n4[i]*n9[i])/2. + (n5[i]*n9[i])/4.)
        denom[i] = 1.*n[i]*(n[i]-1)
    return 2. * numer / denom     ### check factor of four

def compute_D2(counts):
    n1 = counts[:,0]
    n2 = counts[:,1]
    n3 = counts[:,2]
    n4 = counts[:,3]
    n5 = counts[:,4]
    n6 = counts[:,5]
    n7 = counts[:,6]
    n8 = counts[:,7]
    n9 = counts[:,8]
    n = np.sum(counts, axis=1)
    denom = np.zeros(n.shape)
    numer = np.zeros(n.shape)
    for i in range(len(n)):
        numer[i] = (n2[i]*n4[i] - n2[i]**2*n4[i] + 4*n3[i]*n4[i] - 4*n2[i]*n3[i]*n4[i] - 4*n3[i]**2*n4[i] - n2[i]*n4[i]**2 - 4*n3[i]*n4[i]**2 + n1[i]*n5[i] - n1[i]**2*n5[i] + n3[i]*n5[i] + 2*n1[i]*n3[i]*n5[i] - n3[i]**2*n5[i] - 4*n3[i]*n4[i]*n5[i] - n1[i]*n5[i]**2 - n3[i]*n5[i]**2 + 4*n1[i]*n6[i] - 4*n1[i]**2*n6[i] + n2[i]*n6[i] - 4*n1[i]*n2[i]*n6[i] - n2[i]**2*n6[i] + 2*n2[i]*n4[i]*n6[i] - 4*n1[i]*n5[i]*n6[i] - 4*n1[i]*n6[i]**2 - n2[i]*n6[i]**2 + 4*n2[i]*n7[i] - 4*n2[i]**2*n7[i] + 16*n3[i]*n7[i] - 16*n2[i]*n3[i]*n7[i] - 16*n3[i]**2*n7[i] - 4*n2[i]*n4[i]*n7[i] - 16*n3[i]*n4[i]*n7[i] + n5[i]*n7[i] + 2*n1[i]*n5[i]*n7[i] - 4*n2[i]*n5[i]*n7[i] - 18*n3[i]*n5[i]*n7[i] - n5[i]**2*n7[i] + 4*n6[i]*n7[i] + 8*n1[i]*n6[i]*n7[i] - 16*n3[i]*n6[i]*n7[i] - 4*n5[i]*n6[i]*n7[i] - 4*n6[i]**2*n7[i] - 4*n2[i]*n7[i]**2 - 16*n3[i]*n7[i]**2 - n5[i]*n7[i]**2 - 4*n6[i]*n7[i]**2 + 4*n1[i]*n8[i] - 4*n1[i]**2*n8[i] + 4*n3[i]*n8[i] + 8*n1[i]*n3[i]*n8[i] - 4*n3[i]**2*n8[i] + n4[i]*n8[i] - 4*n1[i]*n4[i]*n8[i] + 2*n2[i]*n4[i]*n8[i] - n4[i]**2*n8[i] - 4*n1[i]*n5[i]*n8[i] - 4*n3[i]*n5[i]*n8[i] + n6[i]*n8[i] + 2*n2[i]*n6[i]*n8[i] - 4*n3[i]*n6[i]*n8[i] + 2*n4[i]*n6[i]*n8[i] - n6[i]**2*n8[i] - 16*n3[i]*n7[i]*n8[i] - 4*n6[i]*n7[i]*n8[i] - 4*n1[i]*n8[i]**2 - 4*n3[i]*n8[i]**2 - n4[i]*n8[i]**2 - n6[i]*n8[i]**2 + 16*n1[i]*n9[i] - 16*n1[i]**2*n9[i] + 4*n2[i]*n9[i] - 16*n1[i]*n2[i]*n9[i] - 4*n2[i]**2*n9[i] + 4*n4[i]*n9[i] - 16*n1[i]*n4[i]*n9[i] + 8*n3[i]*n4[i]*n9[i] - 4*n4[i]**2*n9[i] + n5[i]*n9[i] - 18*n1[i]*n5[i]*n9[i] - 4*n2[i]*n5[i]*n9[i] + 2*n3[i]*n5[i]*n9[i] - 4*n4[i]*n5[i]*n9[i] - n5[i]**2*n9[i] - 16*n1[i]*n6[i]*n9[i] - 4*n2[i]*n6[i]*n9[i] + 8*n2[i]*n7[i]*n9[i] + 2*n5[i]*n7[i]*n9[i] - 16*n1[i]*n8[i]*n9[i] - 4*n4[i]*n8[i]*n9[i] - 16*n1[i]*n9[i]**2 - 4*n2[i]*n9[i]**2 - 4*n4[i]*n9[i]**2 - n5[i]*n9[i]**2)/16. + (-((n2[i]/2. + n3[i] + n5[i]/4. + n6[i]/2.)*(n4[i]/2. + n5[i]/4. + n7[i] + n8[i]/2.)) + (n1[i] + n2[i]/2. + n4[i]/2. + n5[i]/4.)*(n5[i]/4. + n6[i]/2. + n8[i]/2. + n9[i]))**2
        denom[i] = 1.*n[i]*(n[i]-1)*(n[i]-2)*(n[i]-3)
    return 4. * numer / denom     ### check factor of four

def compute_Dz(counts):
    n1 = counts[:,0]
    n2 = counts[:,1]
    n3 = counts[:,2]
    n4 = counts[:,3]
    n5 = counts[:,4]
    n6 = counts[:,5]
    n7 = counts[:,6]
    n8 = counts[:,7]
    n9 = counts[:,8]
    n = np.sum(counts, axis=1)
    denom = np.zeros(n.shape)
    numer = np.zeros(n.shape)
    for i in range(len(n)):
        numer[i] = (-(n2[i]*n4[i]) + 3*n1[i]*n2[i]*n4[i] + n2[i]**2*n4[i] + 2*n3[i]*n4[i] + 4*n1[i]*n3[i]*n4[i] - n2[i]*n3[i]*n4[i] - 4*n3[i]**2*n4[i] + n2[i]*n4[i]**2 + 2*n3[i]*n4[i]**2 + 2*n1[i]*n5[i] - 3*n1[i]**2*n5[i] - n1[i]*n2[i]*n5[i] + 2*n3[i]*n5[i] + 2*n1[i]*n3[i]*n5[i] - n2[i]*n3[i]*n5[i] - 3*n3[i]**2*n5[i] - n1[i]*n4[i]*n5[i] + n3[i]*n4[i]*n5[i] + 2*n1[i]*n6[i] - 4*n1[i]**2*n6[i] - n2[i]*n6[i] - n1[i]*n2[i]*n6[i] + n2[i]**2*n6[i] + 4*n1[i]*n3[i]*n6[i] + 3*n2[i]*n3[i]*n6[i] - 2*n1[i]*n4[i]*n6[i] - 2*n2[i]*n4[i]*n6[i] - 2*n3[i]*n4[i]*n6[i] + n1[i]*n5[i]*n6[i] - n3[i]*n5[i]*n6[i] + 2*n1[i]*n6[i]**2 + n2[i]*n6[i]**2 + 2*n2[i]*n7[i] + 4*n1[i]*n2[i]*n7[i] + 2*n2[i]**2*n7[i] + 8*n3[i]*n7[i] + 4*n1[i]*n3[i]*n7[i] - 4*n3[i]**2*n7[i] - n2[i]*n4[i]*n7[i] + 2*n5[i]*n7[i] + 2*n1[i]*n5[i]*n7[i] + n2[i]*n5[i]*n7[i] + 2*n3[i]*n5[i]*n7[i] - n4[i]*n5[i]*n7[i] + 2*n6[i]*n7[i] - n2[i]*n6[i]*n7[i] - 2*n4[i]*n6[i]*n7[i] + n5[i]*n6[i]*n7[i] + 2*n6[i]**2*n7[i] - 4*n2[i]*n7[i]**2 - 4*n3[i]*n7[i]**2 - 3*n5[i]*n7[i]**2 - 4*n6[i]*n7[i]**2 + 2*n1[i]*n8[i] - 4*n1[i]**2*n8[i] - 2*n1[i]*n2[i]*n8[i] + 2*n3[i]*n8[i] - 2*n2[i]*n3[i]*n8[i] - 4*n3[i]**2*n8[i] - n4[i]*n8[i] - n1[i]*n4[i]*n8[i] - 2*n2[i]*n4[i]*n8[i] - n3[i]*n4[i]*n8[i] + n4[i]**2*n8[i] + n1[i]*n5[i]*n8[i] + n3[i]*n5[i]*n8[i] - n6[i]*n8[i] - n1[i]*n6[i]*n8[i] - 2*n2[i]*n6[i]*n8[i] - n3[i]*n6[i]*n8[i] - 2*n4[i]*n6[i]*n8[i] + n6[i]**2*n8[i] + 4*n1[i]*n7[i]*n8[i] - 2*n2[i]*n7[i]*n8[i] + 3*n4[i]*n7[i]*n8[i] - n5[i]*n7[i]*n8[i] - n6[i]*n7[i]*n8[i] + 2*n1[i]*n8[i]**2 + 2*n3[i]*n8[i]**2 + n4[i]*n8[i]**2 + n6[i]*n8[i]**2 + 8*n1[i]*n9[i] - 4*n1[i]**2*n9[i] + 2*n2[i]*n9[i] + 2*n2[i]**2*n9[i] + 4*n1[i]*n3[i]*n9[i] + 4*n2[i]*n3[i]*n9[i] + 2*n4[i]*n9[i] - n2[i]*n4[i]*n9[i] + 2*n4[i]**2*n9[i] + 2*n5[i]*n9[i] + 2*n1[i]*n5[i]*n9[i] + n2[i]*n5[i]*n9[i] + 2*n3[i]*n5[i]*n9[i] + n4[i]*n5[i]*n9[i] - n2[i]*n6[i]*n9[i] - 2*n4[i]*n6[i]*n9[i] - n5[i]*n6[i]*n9[i] + 4*n1[i]*n7[i]*n9[i] + 4*n3[i]*n7[i]*n9[i] + 4*n4[i]*n7[i]*n9[i] + 2*n5[i]*n7[i]*n9[i] + 4*n6[i]*n7[i]*n9[i] - 2*n2[i]*n8[i]*n9[i] + 4*n3[i]*n8[i]*n9[i] - n4[i]*n8[i]*n9[i] - n5[i]*n8[i]*n9[i] + 3*n6[i]*n8[i]*n9[i] - 4*n1[i]*n9[i]**2 - 4*n2[i]*n9[i]**2 - 4*n4[i]*n9[i]**2 - 3*n5[i]*n9[i]**2)/4. + (-n1[i] + n3[i] - n4[i] + n6[i] - n7[i] + n9[i])*(-n1[i] - n2[i] - n3[i] + n7[i] + n8[i] + n9[i])*(-((n2[i]/2. + n3[i] + n5[i]/4. + n6[i]/2.)*(n4[i]/2. + n5[i]/4. + n7[i] + n8[i]/2.)) + (n1[i] + n2[i]/2. + n4[i]/2. + n5[i]/4.)*(n5[i]/4. + n6[i]/2. + n8[i]/2. + n9[i]))
        denom[i] = 1.*n[i]*(n[i]-1)*(n[i]-2)*(n[i]-3)
    return 2. * numer / denom     ### check factor of four

def compute_pi2(counts):
    n1 = counts[:,0]
    n2 = counts[:,1]
    n3 = counts[:,2]
    n4 = counts[:,3]
    n5 = counts[:,4]
    n6 = counts[:,5]
    n7 = counts[:,6]
    n8 = counts[:,7]
    n9 = counts[:,8]
    n = np.sum(counts, axis=1)
    denom = np.zeros(n.shape)
    numer = np.zeros(n.shape)

    for i in range(len(n)):
        numer[i] = (n1[i] + n2[i] + n3[i] + n4[i]/2. + n5[i]/2. + n6[i]/2.)*(n1[i] + n2[i]/2. + n4[i] + n5[i]/2. + n7[i] + n8[i]/2.)*(n2[i]/2. + n3[i] + n5[i]/2. + n6[i] + n8[i]/2. + n9[i])*(n4[i]/2. + n5[i]/2. + n6[i]/2. + n7[i] + n8[i] + n9[i]) + (13*n2[i]*n4[i] - 16*n1[i]*n2[i]*n4[i] - 11*n2[i]**2*n4[i] + 16*n3[i]*n4[i] - 28*n1[i]*n3[i]*n4[i] - 24*n2[i]*n3[i]*n4[i] - 8*n3[i]**2*n4[i] - 11*n2[i]*n4[i]**2 - 20*n3[i]*n4[i]**2 - 6*n5[i] + 12*n1[i]*n5[i] - 4*n1[i]**2*n5[i] + 17*n2[i]*n5[i] - 20*n1[i]*n2[i]*n5[i] - 11*n2[i]**2*n5[i] + 12*n3[i]*n5[i] - 28*n1[i]*n3[i]*n5[i] - 20*n2[i]*n3[i]*n5[i] - 4*n3[i]**2*n5[i] + 17*n4[i]*n5[i] - 20*n1[i]*n4[i]*n5[i] - 32*n2[i]*n4[i]*n5[i] - 40*n3[i]*n4[i]*n5[i] - 11*n4[i]**2*n5[i] + 11*n5[i]**2 - 16*n1[i]*n5[i]**2 - 17*n2[i]*n5[i]**2 - 16*n3[i]*n5[i]**2 - 17*n4[i]*n5[i]**2 - 6*n5[i]**3 + 16*n1[i]*n6[i] - 8*n1[i]**2*n6[i] + 13*n2[i]*n6[i] - 24*n1[i]*n2[i]*n6[i] - 11*n2[i]**2*n6[i] - 28*n1[i]*n3[i]*n6[i] - 16*n2[i]*n3[i]*n6[i] + 24*n4[i]*n6[i] - 36*n1[i]*n4[i]*n6[i] - 38*n2[i]*n4[i]*n6[i] - 36*n3[i]*n4[i]*n6[i] - 20*n4[i]**2*n6[i] + 17*n5[i]*n6[i] - 40*n1[i]*n5[i]*n6[i] - 32*n2[i]*n5[i]*n6[i] - 20*n3[i]*n5[i]*n6[i] - 42*n4[i]*n5[i]*n6[i] - 17*n5[i]**2*n6[i] - 20*n1[i]*n6[i]**2 - 11*n2[i]*n6[i]**2 - 20*n4[i]*n6[i]**2 - 11*n5[i]*n6[i]**2 + 16*n2[i]*n7[i] - 28*n1[i]*n2[i]*n7[i] - 20*n2[i]**2*n7[i] + 16*n3[i]*n7[i] - 48*n1[i]*n3[i]*n7[i] - 44*n2[i]*n3[i]*n7[i] - 16*n3[i]**2*n7[i] - 24*n2[i]*n4[i]*n7[i] - 44*n3[i]*n4[i]*n7[i] + 12*n5[i]*n7[i] - 28*n1[i]*n5[i]*n7[i] - 40*n2[i]*n5[i]*n7[i] - 48*n3[i]*n5[i]*n7[i] - 20*n4[i]*n5[i]*n7[i] - 16*n5[i]**2*n7[i] + 16*n6[i]*n7[i] - 48*n1[i]*n6[i]*n7[i] - 48*n2[i]*n6[i]*n7[i] - 44*n3[i]*n6[i]*n7[i] - 36*n4[i]*n6[i]*n7[i] - 40*n5[i]*n6[i]*n7[i] - 20*n6[i]**2*n7[i] - 8*n2[i]*n7[i]**2 - 16*n3[i]*n7[i]**2 - 4*n5[i]*n7[i]**2 - 8*n6[i]*n7[i]**2 + 16*n1[i]*n8[i] - 8*n1[i]**2*n8[i] + 24*n2[i]*n8[i] - 36*n1[i]*n2[i]*n8[i] - 20*n2[i]**2*n8[i] + 16*n3[i]*n8[i] - 48*n1[i]*n3[i]*n8[i] - 36*n2[i]*n3[i]*n8[i] - 8*n3[i]**2*n8[i] + 13*n4[i]*n8[i] - 24*n1[i]*n4[i]*n8[i] - 38*n2[i]*n4[i]*n8[i] - 48*n3[i]*n4[i]*n8[i] - 11*n4[i]**2*n8[i] + 17*n5[i]*n8[i] - 40*n1[i]*n5[i]*n8[i] - 42*n2[i]*n5[i]*n8[i] - 40*n3[i]*n5[i]*n8[i] - 32*n4[i]*n5[i]*n8[i] - 17*n5[i]**2*n8[i] + 13*n6[i]*n8[i] - 48*n1[i]*n6[i]*n8[i] - 38*n2[i]*n6[i]*n8[i] - 24*n3[i]*n6[i]*n8[i] - 38*n4[i]*n6[i]*n8[i] - 32*n5[i]*n6[i]*n8[i] - 11*n6[i]**2*n8[i] - 28*n1[i]*n7[i]*n8[i] - 36*n2[i]*n7[i]*n8[i] - 44*n3[i]*n7[i]*n8[i] - 16*n4[i]*n7[i]*n8[i] - 20*n5[i]*n7[i]*n8[i] - 24*n6[i]*n7[i]*n8[i] - 20*n1[i]*n8[i]**2 - 20*n2[i]*n8[i]**2 - 20*n3[i]*n8[i]**2 - 11*n4[i]*n8[i]**2 - 11*n5[i]*n8[i]**2 - 11*n6[i]*n8[i]**2 + 16*n1[i]*n9[i] - 16*n1[i]**2*n9[i] + 16*n2[i]*n9[i] - 44*n1[i]*n2[i]*n9[i] - 20*n2[i]**2*n9[i] - 48*n1[i]*n3[i]*n9[i] - 28*n2[i]*n3[i]*n9[i] + 16*n4[i]*n9[i] - 44*n1[i]*n4[i]*n9[i] - 48*n2[i]*n4[i]*n9[i] - 48*n3[i]*n4[i]*n9[i] - 20*n4[i]**2*n9[i] + 12*n5[i]*n9[i] - 48*n1[i]*n5[i]*n9[i] - 40*n2[i]*n5[i]*n9[i] - 28*n3[i]*n5[i]*n9[i] - 40*n4[i]*n5[i]*n9[i] - 16*n5[i]**2*n9[i] - 44*n1[i]*n6[i]*n9[i] - 24*n2[i]*n6[i]*n9[i] - 36*n4[i]*n6[i]*n9[i] - 20*n5[i]*n6[i]*n9[i] - 48*n1[i]*n7[i]*n9[i] - 48*n2[i]*n7[i]*n9[i] - 48*n3[i]*n7[i]*n9[i] - 28*n4[i]*n7[i]*n9[i] - 28*n5[i]*n7[i]*n9[i] - 28*n6[i]*n7[i]*n9[i] - 44*n1[i]*n8[i]*n9[i] - 36*n2[i]*n8[i]*n9[i] - 28*n3[i]*n8[i]*n9[i] - 24*n4[i]*n8[i]*n9[i] - 20*n5[i]*n8[i]*n9[i] - 16*n6[i]*n8[i]*n9[i] - 16*n1[i]*n9[i]**2 - 8*n2[i]*n9[i]**2 - 8*n4[i]*n9[i]**2 - 4*n5[i]*n9[i]**2)/16.
        denom[i] = 1.*n[i]*(n[i]-1)*(n[i]-2)*(n[i]-3)
    return numer / denom

def compute_pi2_1pop(counts):
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = counts
    n = np.sum(counts)
    numer = (
                ((n1 + n2 + n3 + n4/2. + n5/2. + n6/2.)*(n1 + n2/2. + n4 + n5/2. + n7 + n8/2.) *
                -(n2/2. + n3 + n5/2. + n6 + n8/2. + n9)*(n4/2. + n5/2. + n6/2. + n7 + n8 + n9)) +
                ((13*n2*n4 - 16*n1*n2*n4 - 11*n2**2*n4 + 16*n3*n4 - 28*n1*n3*n4 - 24*n2*n3*n4 - 8*n3**2*n4 - 
                    - 11*n2*n4**2 - 20*n3*n4**2 - 6*n5 + 12*n1*n5 - 4*n1**2*n5 + 17*n2*n5 - 20*n1*n2*n5 - 
                    - 11*n2**2*n5 + 12*n3*n5 - 28*n1*n3*n5 - 20*n2*n3*n5 - 4*n3**2*n5 + 17*n4*n5 - 20*n1*n4*n5 - 
                    - 32*n2*n4*n5 - 40*n3*n4*n5 - 11*n4**2*n5 + 11*n5**2 - 16*n1*n5**2 - 17*n2*n5**2 - 
                    - 16*n3*n5**2 - 17*n4*n5**2 - 6*n5**3 + 16*n1*n6 - 8*n1**2*n6 + 13*n2*n6 - 24*n1*n2*n6 - 
                    - 11*n2**2*n6 - 28*n1*n3*n6 - 16*n2*n3*n6 + 24*n4*n6 - 36*n1*n4*n6 - 38*n2*n4*n6 - 
                    - 36*n3*n4*n6 - 20*n4**2*n6 + 17*n5*n6 - 40*n1*n5*n6 - 32*n2*n5*n6 - 20*n3*n5*n6 - 
                    - 42*n4*n5*n6 - 17*n5**2*n6 - 20*n1*n6**2 - 11*n2*n6**2 - 20*n4*n6**2 - 11*n5*n6**2 + 
                    - 16*n2*n7 - 28*n1*n2*n7 - 20*n2**2*n7 + 16*n3*n7 - 48*n1*n3*n7 - 44*n2*n3*n7 - 16*n3**2*n7 - 
                    - 24*n2*n4*n7 - 44*n3*n4*n7 + 12*n5*n7 - 28*n1*n5*n7 - 40*n2*n5*n7 - 48*n3*n5*n7 - 
                    - 20*n4*n5*n7 - 16*n5**2*n7 + 16*n6*n7 - 48*n1*n6*n7 - 48*n2*n6*n7 - 44*n3*n6*n7 - 
                    - 36*n4*n6*n7 - 40*n5*n6*n7 - 20*n6**2*n7 - 8*n2*n7**2 - 16*n3*n7**2 - 4*n5*n7**2 - 
                    - 8*n6*n7**2 + 16*n1*n8 - 8*n1**2*n8 + 24*n2*n8 - 36*n1*n2*n8 - 20*n2**2*n8 + 16*n3*n8 - 
                    - 48*n1*n3*n8 - 36*n2*n3*n8 - 8*n3**2*n8 + 13*n4*n8 - 24*n1*n4*n8 - 38*n2*n4*n8 - 
                    - 48*n3*n4*n8 - 11*n4**2*n8 + 17*n5*n8 - 40*n1*n5*n8 - 42*n2*n5*n8 - 40*n3*n5*n8 - 
                    - 32*n4*n5*n8 - 17*n5**2*n8 + 13*n6*n8 - 48*n1*n6*n8 - 38*n2*n6*n8 - 24*n3*n6*n8 - 
                    - 38*n4*n6*n8 - 32*n5*n6*n8 - 11*n6**2*n8 - 28*n1*n7*n8 - 36*n2*n7*n8 - 44*n3*n7*n8 - 
                    - 16*n4*n7*n8 - 20*n5*n7*n8 - 24*n6*n7*n8 - 20*n1*n8**2 - 20*n2*n8**2 - 20*n3*n8**2 - 
                    - 11*n4*n8**2 - 11*n5*n8**2 - 11*n6*n8**2 + 16*n1*n9 - 16*n1**2*n9 + 16*n2*n9 - 44*n1*n2*n9 - 
                    - 20*n2**2*n9 - 48*n1*n3*n9 - 28*n2*n3*n9 + 16*n4*n9 - 44*n1*n4*n9 - 48*n2*n4*n9 - 
                    - 48*n3*n4*n9 - 20*n4**2*n9 + 12*n5*n9 - 48*n1*n5*n9 - 40*n2*n5*n9 - 28*n3*n5*n9 - 
                    - 40*n4*n5*n9 - 16*n5**2*n9 - 44*n1*n6*n9 - 24*n2*n6*n9 - 36*n4*n6*n9 - 20*n5*n6*n9 - 
                    - 48*n1*n7*n9 - 48*n2*n7*n9 - 48*n3*n7*n9 - 28*n4*n7*n9 - 28*n5*n7*n9 - 28*n6*n7*n9 - 
                    - 44*n1*n8*n9 - 36*n2*n8*n9 - 28*n3*n8*n9 - 24*n4*n8*n9 - 20*n5*n8*n9 - 16*n6*n8*n9 - 
                    - 16*n1*n9**2 - 8*n2*n9**2 - 8*n4*n9**2 - 4*n5*n9**2)/16.
                )
    )
    denom = n * (n-1) * (n-2) * (n-3)
    return numer / denom

class GenoCalcTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_compute_D(self):
        test_array = np.random.randint(1, high=100, size=(100,9), dtype=np.int32)
        self.assertTrue(np.allclose(compute_D(test_array), geno_calc.compute_D(test_array)))

    def test_compute_D2(self):
        test_array = np.random.randint(1, high=100, size=(100,9), dtype=np.int32)
        self.assertTrue(np.allclose(compute_D2(test_array), geno_calc.compute_D2(test_array)))
    
    def test_compute_Dz(self):
        test_array = np.random.randint(1, high=100, size=(100,9), dtype=np.int32)
        self.assertTrue(np.allclose(compute_Dz(test_array), geno_calc.compute_Dz(test_array)))

    def test_compute_pi2(self):
        test_array = np.random.randint(1, high=100, size=(100,9), dtype=np.int32)
        pi2_test = compute_pi2(test_array)
        self.assertTrue(np.allclose(pi2_test, geno_calc.compute_pi2(test_array)))
        for i, counts in enumerate(test_array):
            pi2_1pop = compute_pi2_1pop
            self.assertTrue(pi2_test[i], pi2_1pop)

def main():
    test_array = np.random.randint(1, high=3, size=(1,9), dtype=np.int32)
    #test_array = np.ones((1,9), dtype=np.int32)
    b = 1
    test_array = b*test_array
    print(test_array)
    print(compute_D(test_array), geno_calc.compute_D(test_array))


suite = unittest.TestLoader().loadTestsFromTestCase(GenoCalcTestCase)
if __name__ == "__main__":
    unittest.main()
