import numpy as np
cimport numpy as np

"""
We computed these using Mathematica, using the following commands

gs = {g1, g2, g3, g4, g5, g6, g7, g8, g9};
ns = {n1, n2, n3, n4, n5, n6, n7, n8, n9};
expG[monome_] := Table[Exponent[monome, gs[[i]]], {i, 1, 9}]
ToMonomesG[t_] := MonomialList[t, gs]

unbiasedMonomialG[term_] := Module[{k, vars, coeff},
  k = expG[term];
  vars = Product[gs[[j]]^k[[j]], {j, 1, 9}];
  coeff = Coefficient[term, vars];
  coeff* Product[Binomial[ns[[i]], k[[i]]], {i, 1, 9}] / Binomial[nd, Total[k]] / Multinomial @@ k
]

unbiasedSample[term_] := Plus @@ (unbiasedMonomialG[#] & /@ ToMonomesG[term])

"""

cpdef DD(np.ndarray[np.int64_t, ndim=3] counts, list pop_nums):
    if len(pop_nums) != 2:
        raise ValueError("Can pass only two population indexes to DD")
    
    cdef int pop1, pop2
    pop1 = pop_nums[0]
    pop2 = pop_nums[1]
    
    cdef np.ndarray[np.int64_t, ndim=2] cs, cs1, cs2
    
    cdef np.ndarray[np.int64_t, ndim=1] n1, n2, n3, n4, n5, n6, n7, n8, n9
    
    cdef np.ndarray[np.int64_t, ndim=1] n11,n12,n13,n14,n15,n16,n17,n18,n19
    cdef np.ndarray[np.int64_t, ndim=1] n21,n22,n23,n24,n25,n26,n27,n28,n29
    
    cdef np.ndarray[np.float_t, ndim=1] numer, denom
    
    cdef np.ndarray[np.int64_t, ndim=1] ns, ns1, ns2
    
    cdef int i
    
    if pop1 == pop2:
        cs = counts[pop1]
        n1,n2,n3,n4,n5,n6,n7,n8,n9 = cs
        
        ns = np.sum(cs, axis=0)
        
        denom = 0.*ns
        numer = 0.*ns
        
        for i in range(len(ns)):
            numer[i] = (n2[i]*n4[i] - n2[i]**2*n4[i] + 4*n3[i]*n4[i] - 4*n2[i]*n3[i]*n4[i] - 4*n3[i]**2*n4[i] - n2[i]*n4[i]**2 - 4*n3[i]*n4[i]**2 + n1[i]*n5[i] - n1[i]**2*n5[i] + n3[i]*n5[i] + 2*n1[i]*n3[i]*n5[i] - n3[i]**2*n5[i] - 4*n3[i]*n4[i]*n5[i] - n1[i]*n5[i]**2 - n3[i]*n5[i]**2 + 4*n1[i]*n6[i] - 4*n1[i]**2*n6[i] + n2[i]*n6[i] - 4*n1[i]*n2[i]*n6[i] - n2[i]**2*n6[i] + 2*n2[i]*n4[i]*n6[i] - 4*n1[i]*n5[i]*n6[i] - 4*n1[i]*n6[i]**2 - n2[i]*n6[i]**2 + 4*n2[i]*n7[i] - 4*n2[i]**2*n7[i] + 16*n3[i]*n7[i] - 16*n2[i]*n3[i]*n7[i] - 16*n3[i]**2*n7[i] - 4*n2[i]*n4[i]*n7[i] - 16*n3[i]*n4[i]*n7[i] + n5[i]*n7[i] + 2*n1[i]*n5[i]*n7[i] - 4*n2[i]*n5[i]*n7[i] - 18*n3[i]*n5[i]*n7[i] - n5[i]**2*n7[i] + 4*n6[i]*n7[i] + 8*n1[i]*n6[i]*n7[i] - 16*n3[i]*n6[i]*n7[i] - 4*n5[i]*n6[i]*n7[i] - 4*n6[i]**2*n7[i] - 4*n2[i]*n7[i]**2 - 16*n3[i]*n7[i]**2 - n5[i]*n7[i]**2 - 4*n6[i]*n7[i]**2 + 4*n1[i]*n8[i] - 4*n1[i]**2*n8[i] + 4*n3[i]*n8[i] + 8*n1[i]*n3[i]*n8[i] - 4*n3[i]**2*n8[i] + n4[i]*n8[i] - 4*n1[i]*n4[i]*n8[i] + 2*n2[i]*n4[i]*n8[i] - n4[i]**2*n8[i] - 4*n1[i]*n5[i]*n8[i] - 4*n3[i]*n5[i]*n8[i] + n6[i]*n8[i] + 2*n2[i]*n6[i]*n8[i] - 4*n3[i]*n6[i]*n8[i] + 2*n4[i]*n6[i]*n8[i] - n6[i]**2*n8[i] - 16*n3[i]*n7[i]*n8[i] - 4*n6[i]*n7[i]*n8[i] - 4*n1[i]*n8[i]**2 - 4*n3[i]*n8[i]**2 - n4[i]*n8[i]**2 - n6[i]*n8[i]**2 + 16*n1[i]*n9[i] - 16*n1[i]**2*n9[i] + 4*n2[i]*n9[i] - 16*n1[i]*n2[i]*n9[i] - 4*n2[i]**2*n9[i] + 4*n4[i]*n9[i] - 16*n1[i]*n4[i]*n9[i] + 8*n3[i]*n4[i]*n9[i] - 4*n4[i]**2*n9[i] + n5[i]*n9[i] - 18*n1[i]*n5[i]*n9[i] - 4*n2[i]*n5[i]*n9[i] + 2*n3[i]*n5[i]*n9[i] - 4*n4[i]*n5[i]*n9[i] - n5[i]**2*n9[i] - 16*n1[i]*n6[i]*n9[i] - 4*n2[i]*n6[i]*n9[i] + 8*n2[i]*n7[i]*n9[i] + 2*n5[i]*n7[i]*n9[i] - 16*n1[i]*n8[i]*n9[i] - 4*n4[i]*n8[i]*n9[i] - 16*n1[i]*n9[i]**2 - 4*n2[i]*n9[i]**2 - 4*n4[i]*n9[i]**2 - n5[i]*n9[i]**2)/16. + (-((n2[i]/2. + n3[i] + n5[i]/4. + n6[i]/2.)*(n4[i]/2. + n5[i]/4. + n7[i] + n8[i]/2.)) + (n1[i] + n2[i]/2. + n4[i]/2. + n5[i]/4.)*(n5[i]/4. + n6[i]/2. + n8[i]/2. + n9[i]))**2
            denom[i] = ns[i]*(ns[i]-1)*(ns[i]-2)*(ns[i]-3)
        return 4. * numer / denom
    else:
        # compute D1 D2 for the two pops in pop_nums
        cs1 = counts[pop1]
        cs2 = counts[pop2]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        
        denom = 0.*ns1
        numer = 0.*ns1
        
        for i in range(len(ns1)):
            numer[i] = ( (-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-((n22[i]/2. + n23[i] + n25[i]/4. + n26[i]/2.)*(n24[i]/2. + n25[i]/4. + n27[i] + n28[i]/2.)) + (n21[i] + n22[i]/2. + n24[i]/2. + n25[i]/4.)*(n25[i]/4. + n26[i]/2. + n28[i]/2. + n29[i])) )
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*(ns2[i]-1)
        return 4. * numer / denom

def Dz(counts, pop_nums):
    if len(pop_nums) != 3:
        raise ValueError("Must pass three populations to Dz")
    cdef int pop1, pop2, pop3
    
    pop1,pop2,pop3 = pop_nums
    
    cdef np.ndarray[np.int64_t, ndim=2] cs, cs1, cs2, cs3
    
    cdef np.ndarray[np.int64_t, ndim=1] n1, n2, n3, n4, n5, n6, n7, n8, n9
    
    cdef np.ndarray[np.int64_t, ndim=1] n11,n12,n13,n14,n15,n16,n17,n18,n19
    cdef np.ndarray[np.int64_t, ndim=1] n21,n22,n23,n24,n25,n26,n27,n28,n29
    cdef np.ndarray[np.int64_t, ndim=1] n31,n32,n33,n34,n35,n36,n37,n38,n39
    
    cdef np.ndarray[np.float_t, ndim=1] numer, denom
    
    cdef np.ndarray[np.int64_t, ndim=1] ns, ns1, ns2, ns3
    
    cdef int i
    
    if pop1 == pop2 == pop3: # Dz(i,i,i)
        cs = counts[pop1]
        n1,n2,n3,n4,n5,n6,n7,n8,n9 = cs
        ns = np.sum(cs, axis=0)
        denom = 0.*ns
        numer = 0.*ns
        for i in range(len(ns)):
            numer[i] = (-(n2[i]*n4[i]) + 3*n1[i]*n2[i]*n4[i] + n2[i]**2*n4[i] + 2*n3[i]*n4[i] + 4*n1[i]*n3[i]*n4[i] - n2[i]*n3[i]*n4[i] - 4*n3[i]**2*n4[i] + n2[i]*n4[i]**2 + 2*n3[i]*n4[i]**2 + 2*n1[i]*n5[i] - 3*n1[i]**2*n5[i] - n1[i]*n2[i]*n5[i] + 2*n3[i]*n5[i] + 2*n1[i]*n3[i]*n5[i] - n2[i]*n3[i]*n5[i] - 3*n3[i]**2*n5[i] - n1[i]*n4[i]*n5[i] + n3[i]*n4[i]*n5[i] + 2*n1[i]*n6[i] - 4*n1[i]**2*n6[i] - n2[i]*n6[i] - n1[i]*n2[i]*n6[i] + n2[i]**2*n6[i] + 4*n1[i]*n3[i]*n6[i] + 3*n2[i]*n3[i]*n6[i] - 2*n1[i]*n4[i]*n6[i] - 2*n2[i]*n4[i]*n6[i] - 2*n3[i]*n4[i]*n6[i] + n1[i]*n5[i]*n6[i] - n3[i]*n5[i]*n6[i] + 2*n1[i]*n6[i]**2 + n2[i]*n6[i]**2 + 2*n2[i]*n7[i] + 4*n1[i]*n2[i]*n7[i] + 2*n2[i]**2*n7[i] + 8*n3[i]*n7[i] + 4*n1[i]*n3[i]*n7[i] - 4*n3[i]**2*n7[i] - n2[i]*n4[i]*n7[i] + 2*n5[i]*n7[i] + 2*n1[i]*n5[i]*n7[i] + n2[i]*n5[i]*n7[i] + 2*n3[i]*n5[i]*n7[i] - n4[i]*n5[i]*n7[i] + 2*n6[i]*n7[i] - n2[i]*n6[i]*n7[i] - 2*n4[i]*n6[i]*n7[i] + n5[i]*n6[i]*n7[i] + 2*n6[i]**2*n7[i] - 4*n2[i]*n7[i]**2 - 4*n3[i]*n7[i]**2 - 3*n5[i]*n7[i]**2 - 4*n6[i]*n7[i]**2 + 2*n1[i]*n8[i] - 4*n1[i]**2*n8[i] - 2*n1[i]*n2[i]*n8[i] + 2*n3[i]*n8[i] - 2*n2[i]*n3[i]*n8[i] - 4*n3[i]**2*n8[i] - n4[i]*n8[i] - n1[i]*n4[i]*n8[i] - 2*n2[i]*n4[i]*n8[i] - n3[i]*n4[i]*n8[i] + n4[i]**2*n8[i] + n1[i]*n5[i]*n8[i] + n3[i]*n5[i]*n8[i] - n6[i]*n8[i] - n1[i]*n6[i]*n8[i] - 2*n2[i]*n6[i]*n8[i] - n3[i]*n6[i]*n8[i] - 2*n4[i]*n6[i]*n8[i] + n6[i]**2*n8[i] + 4*n1[i]*n7[i]*n8[i] - 2*n2[i]*n7[i]*n8[i] + 3*n4[i]*n7[i]*n8[i] - n5[i]*n7[i]*n8[i] - n6[i]*n7[i]*n8[i] + 2*n1[i]*n8[i]**2 + 2*n3[i]*n8[i]**2 + n4[i]*n8[i]**2 + n6[i]*n8[i]**2 + 8*n1[i]*n9[i] - 4*n1[i]**2*n9[i] + 2*n2[i]*n9[i] + 2*n2[i]**2*n9[i] + 4*n1[i]*n3[i]*n9[i] + 4*n2[i]*n3[i]*n9[i] + 2*n4[i]*n9[i] - n2[i]*n4[i]*n9[i] + 2*n4[i]**2*n9[i] + 2*n5[i]*n9[i] + 2*n1[i]*n5[i]*n9[i] + n2[i]*n5[i]*n9[i] + 2*n3[i]*n5[i]*n9[i] + n4[i]*n5[i]*n9[i] - n2[i]*n6[i]*n9[i] - 2*n4[i]*n6[i]*n9[i] - n5[i]*n6[i]*n9[i] + 4*n1[i]*n7[i]*n9[i] + 4*n3[i]*n7[i]*n9[i] + 4*n4[i]*n7[i]*n9[i] + 2*n5[i]*n7[i]*n9[i] + 4*n6[i]*n7[i]*n9[i] - 2*n2[i]*n8[i]*n9[i] + 4*n3[i]*n8[i]*n9[i] - n4[i]*n8[i]*n9[i] - n5[i]*n8[i]*n9[i] + 3*n6[i]*n8[i]*n9[i] - 4*n1[i]*n9[i]**2 - 4*n2[i]*n9[i]**2 - 4*n4[i]*n9[i]**2 - 3*n5[i]*n9[i]**2)/4. + (-n1[i] + n3[i] - n4[i] + n6[i] - n7[i] + n9[i])*(-n1[i] - n2[i] - n3[i] + n7[i] + n8[i] + n9[i])*(-((n2[i]/2. + n3[i] + n5[i]/4. + n6[i]/2.)*(n4[i]/2. + n5[i]/4. + n7[i] + n8[i]/2.)) + (n1[i] + n2[i]/2. + n4[i]/2. + n5[i]/4.)*(n5[i]/4. + n6[i]/2. + n8[i]/2. + n9[i]))
            denom[i] = ns[i]*(ns[i]-1)*(ns[i]-2)*(ns[i]-3)
        return 2. * numer / denom
    elif pop1 == pop2: # Dz(i,i,j)
        cs1,cs2 = counts[pop1],counts[pop3]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] =  ((2*n12[i]*n17[i] + 4*n13[i]*n17[i] + 2*n15[i]*n17[i] + 4*n16[i]*n17[i] - 2*n11[i]*n18[i] + 2*n13[i]*n18[i] - 2*n14[i]*n18[i] + 2*n16[i]*n18[i] - 4*n11[i]*n19[i] - 2*n12[i]*n19[i] - 4*n14[i]*n19[i] - 2*n15[i]*n19[i])*(-n21[i] + n23[i] - n24[i] + n26[i] - n27[i] + n29[i]))/4. + (-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] + n23[i] - n24[i] + n26[i] - n27[i] + n29[i]) + (-n11[i] - n12[i] - n13[i] + n17[i] + n18[i] + n19[i])*(-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] + n23[i] - n24[i] + n26[i] - n27[i] + n29[i])
            denom[i] = ns2[i] * ns1[i] * (ns1[i] - 1.) * (ns1[i] - 2)
        return 2. * numer / denom
    elif pop1 == pop3: # Dz(i,j,i)
        cs1,cs2 = counts[pop1],counts[pop2]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = ((2*n13[i]*n14[i] + 2*n13[i]*n15[i] - 2*n11[i]*n16[i] - 2*n12[i]*n16[i] + 4*n13[i]*n17[i] + 2*n16[i]*n17[i] + 4*n13[i]*n18[i] + 2*n16[i]*n18[i] - 4*n11[i]*n19[i] - 4*n12[i]*n19[i] - 2*n14[i]*n19[i] - 2*n15[i]*n19[i])*(-n21[i] - n22[i] - n23[i] + n27[i] + n28[i] + n29[i]))/4. + (-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] - n22[i] - n23[i] + n27[i] + n28[i] + n29[i]) + (-n11[i] + n13[i] - n14[i] + n16[i] - n17[i] + n19[i])*(-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] - n22[i] - n23[i] + n27[i] + n28[i] + n29[i])
            denom[i] = ns2[i] * ns1[i] * (ns1[i] - 1.) * (ns1[i] - 2)
        return 2. * numer / denom
    elif pop2 == pop3: # Dz(i,j,j)
        cs1,cs2 = counts[pop1],counts[pop2]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = (-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] + n23[i] + n27[i] - n29[i]) + (-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] + n23[i] - n24[i] + n26[i] - n27[i] + n29[i])*(-n21[i] - n22[i] - n23[i] + n27[i] + n28[i] + n29[i])
            denom[i] = ns2[i] * (ns2[i]-1) * ns1[i] * (ns1[i] - 1.)
        return 2. * numer / denom
    else: # Dz(i,j,k)
        cs1 = counts[pop1]
        cs2 = counts[pop2]
        cs3 = counts[pop3]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        n31,n32,n33,n34,n35,n36,n37,n38,n39 = cs3
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        ns3 = np.sum(cs3, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = (-((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n14[i]/2. + n15[i]/4. + n17[i] + n18[i]/2.)) + (n11[i] + n12[i]/2. + n14[i]/2. + n15[i]/4.)*(n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i]))*(-n21[i] - n22[i] - n23[i] + n27[i] + n28[i] + n29[i])*(-n31[i] + n33[i] - n34[i] + n36[i] - n37[i] + n39[i])
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*ns3[i]
        return 2. * numer / denom

def pi2(counts, pop_nums):
    if len(pop_nums) != 4:
        raise ValueError("Must pass four populations to pi2")
    cdef int pop1, pop2, pop3
    
    pop1,pop2,pop3,pop4 = pop_nums
    
    cdef np.ndarray[np.int64_t, ndim=2] cs, cs1, cs2, cs3, cs4
    
    cdef np.ndarray[np.int64_t, ndim=1] n1, n2, n3, n4, n5, n6, n7, n8, n9
    
    cdef np.ndarray[np.int64_t, ndim=1] n11,n12,n13,n14,n15,n16,n17,n18,n19
    cdef np.ndarray[np.int64_t, ndim=1] n21,n22,n23,n24,n25,n26,n27,n28,n29
    cdef np.ndarray[np.int64_t, ndim=1] n31,n32,n33,n34,n35,n36,n37,n38,n39
    cdef np.ndarray[np.int64_t, ndim=1] n41,n42,n43,n44,n45,n46,n47,n48,n49
    
    cdef np.ndarray[np.float_t, ndim=1] numer, denom
    
    cdef np.ndarray[np.int64_t, ndim=1] ns, ns1, ns2, ns3, ns4
    
    cdef int i

    if pop1 == pop2 == pop3 == pop4: # pi2(i,i;i,i)
        cs = counts[pop1]
        n1,n2,n3,n4,n5,n6,n7,n8,n9 = cs
        ns = np.sum(cs, axis=0)
        denom = 0.*ns
        numer = 0.*ns
        for i in range(len(ns)):
            numer[i] = ((n1[i] + n2[i] + n3[i] + n4[i]/2. + n5[i]/2. + n6[i]/2.)*(n1[i] + n2[i]/2. + n4[i] + n5[i]/2. + n7[i] + n8[i]/2.) *
                        (n2[i]/2. + n3[i] + n5[i]/2. + n6[i] + n8[i]/2. + n9[i])*(n4[i]/2. + n5[i]/2. + n6[i]/2. + n7[i] + n8[i] + n9[i]) + 
                        ((13*n2[i]*n4[i] - 16*n1[i]*n2[i]*n4[i] - 11*n2[i]**2*n4[i] + 16*n3[i]*n4[i] - 28*n1[i]*n3[i]*n4[i] - 24*n2[i]*n3[i]*n4[i]) + 
                        (-8*n3[i]**2*n4[i] - 11*n2[i]*n4[i]**2 - 20*n3[i]*n4[i]**2 - 6*n5[i] + 12*n1[i]*n5[i] - 4*n1[i]**2*n5[i] + 17*n2[i]*n5[i]) + 
                        (-20*n1[i]*n2[i]*n5[i] - 11*n2[i]**2*n5[i] + 12*n3[i]*n5[i] - 28*n1[i]*n3[i]*n5[i] - 20*n2[i]*n3[i]*n5[i] - 4*n3[i]**2*n5[i]) + 
                        (17*n4[i]*n5[i] - 20*n1[i]*n4[i]*n5[i] - 32*n2[i]*n4[i]*n5[i] - 40*n3[i]*n4[i]*n5[i] - 11*n4[i]**2*n5[i] + 11*n5[i]**2) + 
                        (-16*n1[i]*n5[i]**2 - 17*n2[i]*n5[i]**2 - 16*n3[i]*n5[i]**2 - 17*n4[i]*n5[i]**2 - 6*n5[i]**3 + 16*n1[i]*n6[i] - 8*n1[i]**2*n6[i]) + 
                        (13*n2[i]*n6[i] - 24*n1[i]*n2[i]*n6[i] - 11*n2[i]**2*n6[i] - 28*n1[i]*n3[i]*n6[i] - 16*n2[i]*n3[i]*n6[i] + 24*n4[i]*n6[i]) + 
                        (-36*n1[i]*n4[i]*n6[i] - 38*n2[i]*n4[i]*n6[i] - 36*n3[i]*n4[i]*n6[i] - 20*n4[i]**2*n6[i] + 17*n5[i]*n6[i] - 40*n1[i]*n5[i]*n6[i]) + 
                        (-32*n2[i]*n5[i]*n6[i] - 20*n3[i]*n5[i]*n6[i] - 42*n4[i]*n5[i]*n6[i] - 17*n5[i]**2*n6[i] - 20*n1[i]*n6[i]**2 - 11*n2[i]*n6[i]**2) + 
                        (-20*n4[i]*n6[i]**2 - 11*n5[i]*n6[i]**2 + 16*n2[i]*n7[i] - 28*n1[i]*n2[i]*n7[i] - 20*n2[i]**2*n7[i] + 16*n3[i]*n7[i]) + 
                        (-48*n1[i]*n3[i]*n7[i] - 44*n2[i]*n3[i]*n7[i] - 16*n3[i]**2*n7[i] - 24*n2[i]*n4[i]*n7[i] - 44*n3[i]*n4[i]*n7[i]) + 
                        (12*n5[i]*n7[i] - 28*n1[i]*n5[i]*n7[i] - 40*n2[i]*n5[i]*n7[i] - 48*n3[i]*n5[i]*n7[i] - 20*n4[i]*n5[i]*n7[i] - 16*n5[i]**2*n7[i]) + 
                        (16*n6[i]*n7[i] - 48*n1[i]*n6[i]*n7[i] - 48*n2[i]*n6[i]*n7[i] - 44*n3[i]*n6[i]*n7[i] - 36*n4[i]*n6[i]*n7[i] - 40*n5[i]*n6[i]*n7[i]) + 
                        (-20*n6[i]**2*n7[i] - 8*n2[i]*n7[i]**2 - 16*n3[i]*n7[i]**2 - 4*n5[i]*n7[i]**2 - 8*n6[i]*n7[i]**2 + 16*n1[i]*n8[i] - 8*n1[i]**2*n8[i]) + 
                        (24*n2[i]*n8[i] - 36*n1[i]*n2[i]*n8[i] - 20*n2[i]**2*n8[i] + 16*n3[i]*n8[i] - 48*n1[i]*n3[i]*n8[i] - 36*n2[i]*n3[i]*n8[i] - 8*n3[i]**2*n8[i]) + 
                        (13*n4[i]*n8[i] - 24*n1[i]*n4[i]*n8[i] - 38*n2[i]*n4[i]*n8[i] - 48*n3[i]*n4[i]*n8[i] - 11*n4[i]**2*n8[i] + 17*n5[i]*n8[i] - 40*n1[i]*n5[i]*n8[i]) + 
                        (-42*n2[i]*n5[i]*n8[i] - 40*n3[i]*n5[i]*n8[i] - 32*n4[i]*n5[i]*n8[i] - 17*n5[i]**2*n8[i] + 13*n6[i]*n8[i] - 48*n1[i]*n6[i]*n8[i]) + 
                        (-38*n2[i]*n6[i]*n8[i] - 24*n3[i]*n6[i]*n8[i] - 38*n4[i]*n6[i]*n8[i] - 32*n5[i]*n6[i]*n8[i] - 11*n6[i]**2*n8[i] - 28*n1[i]*n7[i]*n8[i]) + 
                        (-36*n2[i]*n7[i]*n8[i] - 44*n3[i]*n7[i]*n8[i] - 16*n4[i]*n7[i]*n8[i] - 20*n5[i]*n7[i]*n8[i] - 24*n6[i]*n7[i]*n8[i] - 20*n1[i]*n8[i]**2) + 
                        (20*n2[i]*n8[i]**2 - 20*n3[i]*n8[i]**2 - 11*n4[i]*n8[i]**2 - 11*n5[i]*n8[i]**2 - 11*n6[i]*n8[i]**2 + 16*n1[i]*n9[i] - 16*n1[i]**2*n9[i]) + 
                        (16*n2[i]*n9[i] - 44*n1[i]*n2[i]*n9[i] - 20*n2[i]**2*n9[i] - 48*n1[i]*n3[i]*n9[i] - 28*n2[i]*n3[i]*n9[i] + 16*n4[i]*n9[i]) + 
                        (-44*n1[i]*n4[i]*n9[i] - 48*n2[i]*n4[i]*n9[i] - 48*n3[i]*n4[i]*n9[i] - 20*n4[i]**2*n9[i] + 12*n5[i]*n9[i] - 48*n1[i]*n5[i]*n9[i]) + 
                        (-40*n2[i]*n5[i]*n9[i] - 28*n3[i]*n5[i]*n9[i] - 40*n4[i]*n5[i]*n9[i] - 16*n5[i]**2*n9[i] - 44*n1[i]*n6[i]*n9[i] - 24*n2[i]*n6[i]*n9[i]) + 
                        (-36*n4[i]*n6[i]*n9[i] - 20*n5[i]*n6[i]*n9[i] - 48*n1[i]*n7[i]*n9[i] - 48*n2[i]*n7[i]*n9[i] - 48*n3[i]*n7[i]*n9[i] - 28*n4[i]*n7[i]*n9[i]) + 
                        (-28*n5[i]*n7[i]*n9[i] - 28*n6[i]*n7[i]*n9[i] - 44*n1[i]*n8[i]*n9[i] - 36*n2[i]*n8[i]*n9[i] - 28*n3[i]*n8[i]*n9[i] - 24*n4[i]*n8[i]*n9[i]) + 
                        (-20*n5[i]*n8[i]*n9[i] - 16*n6[i]*n8[i]*n9[i] - 16*n1[i]*n9[i]**2 - 8*n2[i]*n9[i]**2 - 8*n4[i]*n9[i]**2 - 4*n5[i]*n9[i]**2))/16.)
            denom[i] = ns[i]*(ns[i]-1)*(ns[i]-2)*(ns[i]-3)
        return numer / denom
    elif (pop1 == pop2 == pop3) or (pop1 == pop2 == pop4):  # pi2(i,i;i,j) or pi2(i,i;j,i)
        if pop1 == pop3:
            cs1,cs2 = counts[pop1], counts[pop4]
        else:
            cs1,cs2 = counts[pop1], counts[pop3]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] =  (((-3*n14[i])/2. - (3*n15[i])/2. - (3*n16[i])/2. - 2*n17[i] - 2*n18[i] - 2*n19[i])*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*   (n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*   (n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.))/2. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*((-5*n14[i])/2. - (5*n15[i])/2. - (5*n16[i])/2. - 4*n17[i] - 4*n18[i] - 4*n19[i])*   (n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*   (n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/2. + ((n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n15[i]*n21[i] + 2*n16[i]*n21[i] + 2*n18[i]*n21[i] + 4*n19[i]*n21[i] + 2*n22[i] - n12[i]*n22[i] - 2*n13[i]*n22[i] + 2*n14[i]*n22[i] + n15[i]*n22[i] +      4*n17[i]*n22[i] + 3*n18[i]*n22[i] + 2*n19[i]*n22[i] + 4*n23[i] - 2*n12[i]*n23[i] - 4*n13[i]*n23[i] + 4*n14[i]*n23[i] + n15[i]*n23[i] - 2*n16[i]*n23[i] + 8*n17[i]*n23[i] + 4*n18[i]*n23[i] + n15[i]*n24[i] +      2*n16[i]*n24[i] + 2*n18[i]*n24[i] + 4*n19[i]*n24[i] + 2*n25[i] - n12[i]*n25[i] - 2*n13[i]*n25[i] + 2*n14[i]*n25[i] + n15[i]*n25[i] + 4*n17[i]*n25[i] + 3*n18[i]*n25[i] + 2*n19[i]*n25[i] + 4*n26[i] - 2*n12[i]*n26[i] -      4*n13[i]*n26[i] + 4*n14[i]*n26[i] + n15[i]*n26[i] - 2*n16[i]*n26[i] + 8*n17[i]*n26[i] + 4*n18[i]*n26[i] + n15[i]*n27[i] + 2*n16[i]*n27[i] + 2*n18[i]*n27[i] + 4*n19[i]*n27[i] + 2*n28[i] - n12[i]*n28[i] -      2*n13[i]*n28[i] + 2*n14[i]*n28[i] + n15[i]*n28[i] + 4*n17[i]*n28[i] + 3*n18[i]*n28[i] + 2*n19[i]*n28[i] + 4*n29[i] - 2*n12[i]*n29[i] - 4*n13[i]*n29[i] + 4*n14[i]*n29[i] + n15[i]*n29[i] - 2*n16[i]*n29[i] +      8*n17[i]*n29[i] + 4*n18[i]*n29[i]))/8. + (n15[i]*n21[i] + 2*n16[i]*n21[i] - 2*n17[i]*n22[i] - 2*n18[i]*n22[i] - 2*n19[i]*n22[i] - n15[i]*n23[i] - 2*n16[i]*n23[i] - 4*n17[i]*n23[i] - 4*n18[i]*n23[i] -    4*n19[i]*n23[i] + n15[i]*n24[i] + 2*n16[i]*n24[i] - 2*n17[i]*n25[i] - 2*n18[i]*n25[i] - 2*n19[i]*n25[i] - n15[i]*n26[i] - 2*n16[i]*n26[i] - 4*n17[i]*n26[i] - 4*n18[i]*n26[i] - 4*n19[i]*n26[i] + n15[i]*n27[i] +    2*n16[i]*n27[i] - 2*n17[i]*n28[i] - 2*n18[i]*n28[i] - 2*n19[i]*n28[i] - n15[i]*n29[i] - 2*n16[i]*n29[i] - 4*n17[i]*n29[i] - 4*n18[i]*n29[i] - 4*n19[i]*n29[i])/8. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(-(n15[i]*n21[i]) - 2*n16[i]*n21[i] - 2*n18[i]*n21[i] - 4*n19[i]*n21[i] + n15[i]*n23[i] + 2*n16[i]*n23[i] + 2*n18[i]*n23[i] + 4*n19[i]*n23[i] -      n15[i]*n24[i] - 2*n16[i]*n24[i] - 2*n18[i]*n24[i] - 4*n19[i]*n24[i] + n15[i]*n26[i] + 2*n16[i]*n26[i] + 2*n18[i]*n26[i] + 4*n19[i]*n26[i] - n15[i]*n27[i] - 2*n16[i]*n27[i] - 2*n18[i]*n27[i] - 4*n19[i]*n27[i] +      n15[i]*n29[i] + 2*n16[i]*n29[i] + 2*n18[i]*n29[i] + 4*n19[i]*n29[i]))/8.
            denom[i] = ns2[i]*ns1[i]*(ns1[i]-1)*(ns1[i]-2)
        return 1. * numer / denom
    
    elif (pop1 == pop3 == pop4) or (pop2 == pop3 == pop4):  # pi2(i,j;i,i), pi2(j,i;i,i)
        if pop1 == pop3:
            cs1,cs2 = counts[pop1], counts[pop2]
        elif pop2 == pop3:
            cs1,cs2 = counts[pop2], counts[pop1]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = (((-3*n12[i])/2. - 2*n13[i] - (3*n15[i])/2. - 2*n16[i] - (3*n18[i])/2. - 2*n19[i])*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.))/4. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.))/2. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*((-5*n12[i])/2. - 4*n13[i] - (5*n15[i])/2. - 4*n16[i] - (5*n18[i])/2. - 4*n19[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/2. + ((n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n15[i]*n21[i] + 2*n16[i]*n21[i] + 2*n18[i]*n21[i] + 4*n19[i]*n21[i] + n15[i]*n22[i] + 2*n16[i]*n22[i] + 2*n18[i]*n22[i] + 4*n19[i]*n22[i] + n15[i]*n23[i] +   2*n16[i]*n23[i] + 2*n18[i]*n23[i] + 4*n19[i]*n23[i] + 2*n24[i] + 2*n12[i]*n24[i] + 4*n13[i]*n24[i] - n14[i]*n24[i] + n15[i]*n24[i] + 3*n16[i]*n24[i] - 2*n17[i]*n24[i] + 2*n19[i]*n24[i] + 2*n25[i] + 2*n12[i]*n25[i] +   4*n13[i]*n25[i] - n14[i]*n25[i] + n15[i]*n25[i] + 3*n16[i]*n25[i] - 2*n17[i]*n25[i] + 2*n19[i]*n25[i] + 2*n26[i] + 2*n12[i]*n26[i] + 4*n13[i]*n26[i] - n14[i]*n26[i] + n15[i]*n26[i] + 3*n16[i]*n26[i] - 2*n17[i]*n26[i] +   2*n19[i]*n26[i] + 4*n27[i] + 4*n12[i]*n27[i] + 8*n13[i]*n27[i] - 2*n14[i]*n27[i] + n15[i]*n27[i] + 4*n16[i]*n27[i] - 4*n17[i]*n27[i] - 2*n18[i]*n27[i] + 4*n28[i] + 4*n12[i]*n28[i] + 8*n13[i]*n28[i] -   2*n14[i]*n28[i] + n15[i]*n28[i] + 4*n16[i]*n28[i] - 4*n17[i]*n28[i] - 2*n18[i]*n28[i] + 4*n29[i] + 4*n12[i]*n29[i] + 8*n13[i]*n29[i] - 2*n14[i]*n29[i] + n15[i]*n29[i] + 4*n16[i]*n29[i] - 4*n17[i]*n29[i] - 2*n18[i]*n29[i]  ))/8. + (n15[i]*n21[i] + 2*n18[i]*n21[i] + n15[i]*n22[i] + 2*n18[i]*n22[i] + n15[i]*n23[i] + 2*n18[i]*n23[i] - 2*n13[i]*n24[i] - 2*n16[i]*n24[i] - 2*n19[i]*n24[i] - 2*n13[i]*n25[i] - 2*n16[i]*n25[i] - 2*n19[i]*n25[i] - 2*n13[i]*n26[i] - 2*n16[i]*n26[i] - 2*n19[i]*n26[i] - 4*n13[i]*n27[i] - n15[i]*n27[i] - 4*n16[i]*n27[i] - 2*n18[i]*n27[i] - 4*n19[i]*n27[i] - 4*n13[i]*n28[i] - n15[i]*n28[i] - 4*n16[i]*n28[i] - 2*n18[i]*n28[i] - 4*n19[i]*n28[i] - 4*n13[i]*n29[i] - n15[i]*n29[i] - 4*n16[i]*n29[i] - 2*n18[i]*n29[i] - 4*n19[i]*n29[i])/8. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(-(n15[i]*n21[i]) - 2*n16[i]*n21[i] - 2*n18[i]*n21[i] - 4*n19[i]*n21[i] - n15[i]*n22[i] - 2*n16[i]*n22[i] - 2*n18[i]*n22[i] - 4*n19[i]*n22[i] -   n15[i]*n23[i] - 2*n16[i]*n23[i] - 2*n18[i]*n23[i] - 4*n19[i]*n23[i] + n15[i]*n27[i] + 2*n16[i]*n27[i] + 2*n18[i]*n27[i] + 4*n19[i]*n27[i] + n15[i]*n28[i] + 2*n16[i]*n28[i] + 2*n18[i]*n28[i] + 4*n19[i]*n28[i] +   n15[i]*n29[i] + 2*n16[i]*n29[i] + 2*n18[i]*n29[i] + 4*n19[i]*n29[i]))/8.
            denom[i] = ns2[i]*ns1[i]*(ns1[i]-1)*(ns1[i]-2)
        return 1. * numer / denom

    elif pop1 == pop2 and pop3 == pop4:  # pi2(i,i;j,j)
        cs1,cs2 = counts[pop1], counts[pop3]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(-n22[i] - n25[i] - n28[i]))/4. + (n14[i]*n22[i] + n15[i]*n22[i] + n16[i]*n22[i] + n14[i]*n25[i] + n15[i]*n25[i] + n16[i]*n25[i] + n14[i]*n28[i] + n15[i]*n28[i] + n16[i]*n28[i])/16. + ((-n14[i] - n15[i] - n16[i])*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/4. + (n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i])
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*(ns2[i]-1)
        return 1. * numer / denom

    elif (pop1 == pop3 and pop2 == pop4) or (pop1 == pop4 and pop2 == pop3):  # pi2(i,j;i,j) or pi2(i,j;j,i)
        cs1,cs2 = counts[pop1], counts[pop2]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] =   ((-n14[i]/2. - n15[i]/2. - n16[i]/2. - n17[i] - n18[i] - n19[i])*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.))/4. + ((-n15[i]/4. - n16[i]/2. - n18[i]/2. - n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.))/4. + ((n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.))/4. + ((n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(-n22[i]/2. - n23[i] - n25[i]/2. - n26[i] - n28[i]/2. - n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(-n25[i]/4. - n26[i]/2. - n28[i]/2. - n29[i]))/4. + ((n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(-n25[i]/4. - n26[i]/2. - n28[i]/2. - n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n25[i]/4. + n26[i]/2. + n28[i]/2. + n29[i]))/4. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n25[i]/4. + n26[i]/2. + n28[i]/2. + n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(-n14[i]/2. - n15[i]/2. - n16[i]/2. - n17[i] - n18[i] - n19[i])*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/4. + ((1 + n12[i]/2. + n13[i] - n14[i]/2. + n16[i]/2. - n17[i] - n18[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/4. + ((n15[i]/4. + n16[i]/2. + n18[i]/2. + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/4. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i]))/4. + ((n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(1 + n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + ((-n12[i]/2. - n13[i] - n15[i]/4. - n16[i]/2.)*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(-n12[i]/2. - n13[i] - n15[i]/2. - n16[i] - n18[i]/2. - n19[i] - n22[i]/2. - n23[i] - n25[i]/2. - n26[i] - n28[i]/2. - n29[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + ((n12[i]/2. + n13[i] + n15[i]/4. + n16[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i]))/4. + (n15[i]*n21[i] + 2*n16[i]*n21[i] + 2*n18[i]*n21[i] + 4*n19[i]*n21[i] - n15[i]*n23[i] - 2*n16[i]*n23[i] - 2*n18[i]*n23[i] - 4*n19[i]*n23[i] + n11[i]*n25[i] - n13[i]*n25[i] - n15[i]*n25[i] - 2*n16[i]*n25[i] - n17[i]*n25[i] - 2*n18[i]*n25[i] - 3*n19[i]*n25[i] + 2*n11[i]*n26[i] - 2*n13[i]*n26[i] - 2*n15[i]*n26[i] - 4*n16[i]*n26[i] - 2*n17[i]*n26[i] - 4*n18[i]*n26[i] - 6*n19[i]*n26[i] - n15[i]*n27[i] - 2*n16[i]*n27[i] - 2*n18[i]*n27[i] - 4*n19[i]*n27[i] + 2*n11[i]*n28[i] - 2*n13[i]*n28[i] - 2*n15[i]*n28[i] - 4*n16[i]*n28[i] - 2*n17[i]*n28[i] - 4*n18[i]*n28[i] - 6*n19[i]*n28[i] + 4*n11[i]*n29[i] - 4*n13[i]*n29[i] - 3*n15[i]*n29[i] - 6*n16[i]*n29[i] - 4*n17[i]*n29[i] - 6*n18[i]*n29[i] - 8*n19[i]*n29[i])/16. 
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*(ns2[i]-1)
        return 1. * numer / denom

    elif pop1 == pop2:  # pi2(i,i;j,k)
        cs1 = counts[pop1]
        cs2 = counts[pop3]
        cs3 = counts[pop4]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        n31,n32,n33,n34,n35,n36,n37,n38,n39 = cs3
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        ns3 = np.sum(cs3, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = ((-n14[i]/2. - n15[i]/2. - n16[i]/2.)*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i])*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n22[i]/2. + n23[i] + n25[i]/2. + n26[i] + n28[i]/2. + n29[i])*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.))/2. + ((-n14[i]/2. - n15[i]/2. - n16[i]/2.)*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.)*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i]/2. + n24[i] + n25[i]/2. + n27[i] + n28[i]/2.)*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i]))/2.
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*ns3[i]
        return 1. * numer / denom

    elif pop3 == pop4:  # pi2(i,j;k,k)
        cs1 = counts[pop3]
        cs2 = counts[pop1]
        cs3 = counts[pop2]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        n31,n32,n33,n34,n35,n36,n37,n38,n39 = cs3
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        ns3 = np.sum(cs3, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = ((-n12[i]/2. - n15[i]/2. - n18[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n31[i] + n32[i] + n33[i] + n34[i]/2. + n35[i]/2. + n36[i]/2.))/4. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n31[i] + n32[i] + n33[i] + n34[i]/2. + n35[i]/2. + n36[i]/2.))/2. + ((-n12[i]/2. - n15[i]/2. - n18[i]/2.)*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n34[i]/2. + n35[i]/2. + n36[i]/2. + n37[i] + n38[i] + n39[i]))/4. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n34[i]/2. + n35[i]/2. + n36[i]/2. + n37[i] + n38[i] + n39[i]))/2.
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*ns3[i]
        return 1. * numer / denom

    elif (pop1 == pop3) or (pop1 == pop4) or (pop2 == pop3) or (pop2 == pop4):  # pi2(i,j;i,k) or pi2(i,j;k,i) or pi2(i,j;j,k) or pi2(i,j;k,j)
        if pop1 == pop3:   # pi2(i,j;i,k)
            cs1 = counts[pop1]
            cs2 = counts[pop2]
            cs3 = counts[pop4]
        elif pop1 == pop4: # pi2(i,j;k,i)
            cs1 = counts[pop1]
            cs2 = counts[pop2]
            cs3 = counts[pop3]
        elif pop2 == pop3: # pi2(i,j;j,k)
            cs1 = counts[pop2]
            cs2 = counts[pop1]
            cs3 = counts[pop4]
        elif pop2 == pop4: # pi2(i,j;k,j)
            cs1 = counts[pop2]
            cs2 = counts[pop1]
            cs3 = counts[pop3]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        n31,n32,n33,n34,n35,n36,n37,n38,n39 = cs3
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        ns3 = np.sum(cs3, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] =  ((-n15[i]/4. - n16[i]/2. - n18[i]/2. - n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.))/4. + ((n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.))/4. + ((-n12[i]/2. - n13[i] - n15[i]/4. - n16[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n12[i]/2. + n13[i] + n15[i]/2. + n16[i] + n18[i]/2. + n19[i])*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.))/4. + ((-n14[i]/2. - n15[i]/4. - n17[i] - n18[i]/2.)*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i]))/4. + ((n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i]))/4. + ((-n11[i] - n12[i]/2. - n14[i]/2. - n15[i]/4.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n11[i] + n12[i]/2. + n14[i] + n15[i]/2. + n17[i] + n18[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i]))/4.
            denom[i] = ns1[i]*(ns1[i]-1)*ns2[i]*ns3[i]
        return 1. * numer / denom

    else: # pi2(i,j,k,l)
        cs1,cs2,cs3,cs4 = counts[pop1], counts[pop2], counts[pop3], counts[pop4]
        n11,n12,n13,n14,n15,n16,n17,n18,n19 = cs1
        n21,n22,n23,n24,n25,n26,n27,n28,n29 = cs2
        n31,n32,n33,n34,n35,n36,n37,n38,n39 = cs3
        n41,n42,n43,n44,n45,n46,n47,n48,n49 = cs4
        ns1 = np.sum(cs1, axis=0)
        ns2 = np.sum(cs2, axis=0)
        ns3 = np.sum(cs3, axis=0)
        ns4 = np.sum(cs4, axis=0)
        denom = 0.*ns1
        numer = 0.*ns1
        for i in range(len(ns1)):
            numer[i] = ((n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i])*(n41[i] + n42[i]/2. + n44[i] + n45[i]/2. + n47[i] + n48[i]/2.))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n32[i]/2. + n33[i] + n35[i]/2. + n36[i] + n38[i]/2. + n39[i])*(n41[i] + n42[i]/2. + n44[i] + n45[i]/2. + n47[i] + n48[i]/2.))/4. +((n14[i]/2. + n15[i]/2. + n16[i]/2. + n17[i] + n18[i] + n19[i])*(n21[i] + n22[i] + n23[i] + n24[i]/2. + n25[i]/2. + n26[i]/2.)*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.)*(n42[i]/2. + n43[i] + n45[i]/2. + n46[i] + n48[i]/2. + n49[i]))/4. + ((n11[i] + n12[i] + n13[i] + n14[i]/2. + n15[i]/2. + n16[i]/2.)*(n24[i]/2. + n25[i]/2. + n26[i]/2. + n27[i] + n28[i] + n29[i])*(n31[i] + n32[i]/2. + n34[i] + n35[i]/2. + n37[i] + n38[i]/2.)*(n42[i]/2. + n43[i] + n45[i]/2. + n46[i] + n48[i]/2. + n49[i]))/4.
            denom[i] = ns1[i]*ns2[i]*ns3[i]*ns4[i]
        return 1. * numer / denom

