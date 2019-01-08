"""
Contains heterozygosity functions and statistics
"""
import logging
logging.basicConfig()
logger = logging.getLogger('Heterozygosity')

import numpy as np
import math
import copy

import os,sys

def het_names(num_pops):
    Hs = []
    for ii in range(num_pops):
        for jj in range(ii,num_pops):
            Hs.append('H_{0}_{1}'.format(ii+1,jj+1))
    return Hs

def drift(nus):
    num_pops = len(nus)
    D = np.zeros( ( int(num_pops*(num_pops+1)/2), int(num_pops*(num_pops+1)/2) ) )
    c = 0
    for ii in range(num_pops):
        D[c,c] = 1./nus[ii]
        c += (num_pops-ii)
    return D

def migration(mig_mat):
    num_pops = len(mig_mat)
    M = np.zeros( ( int(num_pops*(num_pops+1)/2), int(num_pops*(num_pops+1)/2) ) )
    Hs = het_names(num_pops)
    for ii,H in enumerate(Hs):
        pop1,pop2 = [int(f) for f in H.split('_')[1:]]
        if pop1 == pop2:
            for jj in range(1,num_pops+1):
                if jj == pop1:
                    continue
                else:
                    M[ii,ii] -= 2*mig_mat[jj-1][pop1-1]
                    if pop1 < jj:
                        M[ii,Hs.index('H_{0}_{1}'.format(pop1,jj))] += 2*mig_mat[jj-1][pop1-1]
                    else:
                        M[ii,Hs.index('H_{0}_{1}'.format(jj,pop1))] += 2*mig_mat[jj-1][pop1-1]
        else:
            for jj in range(1,num_pops+1):
                if jj == pop1:
                    continue
                else:
                    M[ii,ii] -= mig_mat[jj-1][pop1-1]
                    if pop2 <= jj:
                        M[ii,Hs.index('H_{0}_{1}'.format(pop2,jj))] += mig_mat[jj-1][pop1-1]
                    else:
                        M[ii,Hs.index('H_{0}_{1}'.format(jj,pop2))] += mig_mat[jj-1][pop1-1]
            for jj in range(1,num_pops+1):
                if jj == pop2:
                    continue
                else:
                    M[ii,ii] -= mig_mat[jj-1][pop2-1]
                    if pop1 <= jj:
                        M[ii,Hs.index('H_{0}_{1}'.format(pop1,jj))] += mig_mat[jj-1][pop2-1]
                    else:
                        M[ii,Hs.index('H_{0}_{1}'.format(jj,pop1))] += mig_mat[jj-1][pop2-1]
    
    return M

def mutation(u, num_pops, frozen=None):
    if frozen is None:
        return 2*u*np.ones(int(num_pops*(num_pops+1)/2))
    else:
        U = np.zeros( int(num_pops*(num_pops+1)/2) )
        c = 0
        for ii in range(num_pops):
            for jj in range(ii,num_pops):
                if frozen[ii] is not True:
                    U[c] += u
                if frozen[jj] is not True:
                    U[c] += u
                c += 1
        return U

def integrate_het(h, nu, T, dt=0.001, theta=0.001, m=None, ism=True, num_pops=1, frozen=None):
    if callable(nu):
        nus = nu(0)
    else:
        nus = [float(nu_pop) for nu_pop in nu]
    
    U = mutation(theta/2., num_pops, frozen=frozen)
    if num_pops > 1 and m is not None:
        if frozen is not None:
            for ii in range(num_pops):
                if frozen[ii] == True:
                    for jj in range(num_pops):
                        m[ii][jj] = 0
                        m[jj][ii] = 0
        M = migration(m)
    else:
        M = 0*U
    
    EYE = np.eye(U.shape[0])
    
    dt_last = dt
    nus_last = nus
    elapsed_T = 0
    while elapsed_T < T:
        if elapsed_T + dt > T:
            dt = T-elapsed_T
        
        if callable(nu):
            nus = nu(elapsed_T+dt/2.)
        
        if dt != dt_last or nus != nus_last or elapsed_T == 0:
            if frozen is not None:
                for ii,val in enumerate(frozen):
                    if val == True:
                        nus[ii] = 1e20
            D = drift(nus)
            if num_pops > 1 and m is not None:
                Ab = -D+M
            else:
                Ab = -D
            Afd = EYE + dt/2.*Ab
            Abd = np.linalg.inv(EYE - dt/2.*Ab)
        
        h = Abd.dot(Afd.dot(h)+dt*U)
        elapsed_T += dt
        dt_last = copy.copy(dt)
        nus_last = copy.copy(nus)

    return h
    
class Het(np.ma.masked_array):
    """
    Represents heterozygosity statistics, both within and between populations.
    
    Represented as an array: [H11, H12, H13, ..., H22, H23, ..., Hnn]
    """
    
    def __new__(subtype, data, mask=np.ma.nomask, num_pops=1,
                dtype=float, copy=True, fill_value=np.nan, keep_mask=True,
                shrink=True, pop_ids=None):
        
        data = np.asanyarray(data)
        
        if mask is np.ma.nomask:
            mask = np.ma.make_mask_none(data.shape)
            
        subarr = np.ma.masked_array(data, mask=mask, dtype=dtype, copy=copy,
                    fill_value=fill_value, keep_mask=keep_mask, shrink=shrink)
        
        subarr = subarr.view(subtype)
                
        if hasattr(data, 'num_pops'):
            subarr.num_pops = data.num_pops
        else:
            subarr.num_pops = num_pops
        
        if hasattr(data, 'pop_ids'):
            subarr.pop_ids = data.pop_ids
        else:
            subarr.pop_ids = pop_ids
        
        return subarr
    
    def __array_finalize__(self, obj):
        if obj is None: 
            return
        np.ma.masked_array.__array_finalize__(self, obj)
        self.num_pops = getattr(obj, 'num_pops', 'unspecified')
        self.pop_ids = getattr(obj, 'pop_ids', 'unspecified')
    
    def __array_wrap__(self, obj):
        result = obj.view(type(self))
        result = np.ma.masked_array.__array_wrap__(self, obj, 
                                                      context=context)
        result.num_pops = self.num_pops
        result.pop_ids = self.pop_ids
        return result
    
    def _update_from(self, obj):
        np.ma.masked_array._update_from(self, obj)
        if hasattr(obj, 'num_pops'):
            self.num_pops = obj.num_pops
        if hasattr(obj, 'pop_ids'):
            self.pop_ids = obj.pop_ids
        
    __array_priority__ = 20
    
    def __repr__(self):
        return 'Het(%s, num_pops=%s)'\
                % (str(self), str(self.num_pops))
    
    def names(self):
        return het_names(self.num_pops)
    
    def split(self, pop_to_split):
        """
         
        """
        if pop_to_split > self.num_pops:
            raise ValueError("population to split larger than number of pops")
        
        h_from = self.data
        h_new = np.empty(int((self.num_pops+1)*(self.num_pops+2)/2))
        c = 0
        for ii in range(1,self.num_pops+2):
            for jj in range(ii,self.num_pops+2):
                if jj == self.num_pops+1:
                    if ii == jj:
                        h_new[c] = h_from[self.names().index('H_{0}_{0}'.format(pop_to_split))]
                    else:
                        if ii <= pop_to_split:
                            h_new[c] = h_from[self.names().index('H_{0}_{1}'.format(ii,pop_to_split))]
                        else:
                            h_new[c] = h_from[self.names().index('H_{0}_{1}'.format(pop_to_split,ii))]
                else:
                    h_new[c] = h_from[self.names().index('H_{0}_{1}'.format(ii,jj))]
                
                c += 1
        
        return Het(h_new, num_pops=self.num_pops+1)
    
    def swap_pops(self, pop1, pop2):
        if pop1 > self.num_pops or pop2 > self.num_pops or pop1 < 1 or pop2 < 1:
            raise ValueError("Invalid population number specified.")
        if pop1 == pop2:
            return self
        else:
            Hs = het_names(self.num_pops)
            h_new = np.zeros(len(Hs))
            pops_old = list(range(1,self.num_pops+1))
            pops_new = list(range(1,self.num_pops+1))
            pops_new[pop1-1] = pop2
            pops_new[pop2-1] = pop1
            d = dict(zip(pops_old, pops_new))
            for ii,H in enumerate(Hs):
                pops_H = [int(p) for p in H.split('_')[1:]]
                pops_H_new = [d.get(p) for p in pops_H]
                H_new = H.split('_')[0] + '_' + '_'.join([str(p) for p in pops_H_new])
                if int(H_new.split('_')[1]) > int(H_new.split('_')[2]):
                    H_new = 'H_{0}_{1}'.format(int(H_new.split('_')[2]), int(H_new.split('_')[1]))
                h_new[ii] = self.data[Hs.index(H_new)]
            return Het(h_new, num_pops=self.num_pops)
            
    def marginalize(self, pops):
        if self.num_pops == 1:
            print("You're trying to marginalize a single population model.")
            return self
        if hasattr(pops,"__len__") == False:
            pops = [pops]

        Hs_from = het_names(self.num_pops)
        Hs_to = het_names(self.num_pops - len(pops))
        h = np.zeros(len(Hs_to))
        count = 0
        for H in Hs_from:
            H_pops = [int(p) for p in H.split('_')[1:]]
            if len(np.intersect1d(pops, H_pops)) == 0:
                h[count] = self[Hs_from.index(H)]
                count += 1
        return Het(h, num_pops=self.num_pops-len(pops))
    
    def admix(self, pop1, pop2, f):
        pass
    
    def pulse_migrate(self, pop_from, pop_to, f):
        pass
    
    # Ensures that when arithmetic is done with LDstats objects,
    # attributes are preserved. For details, see similar code in
    # moments.Spectrum_mod
    for method in ['__add__','__radd__','__sub__','__rsub__','__mul__',
                   '__rmul__','__div__','__rdiv__','__truediv__','__rtruediv__',
                   '__floordiv__','__rfloordiv__','__rpow__','__pow__']:
        exec("""
def %(method)s(self, other):
    if isinstance(other, np.ma.masked_array):
        newdata = self.data.%(method)s (other.data)
        newmask = np.ma.mask_or(self.mask, other.mask)
    else:
        newdata = self.data.%(method)s (other)
        newmask = self.mask
    outLDstats = self.__class__.__new__(self.__class__, newdata, newmask, 
                                   num_pops=self.num_pops)
    return outLDstats
""" % {'method':method})

    # Methods that modify the Spectrum in-place.
    for method in ['__iadd__','__isub__','__imul__','__idiv__',
                   '__itruediv__','__ifloordiv__','__ipow__']:
        exec("""
def %(method)s(self, other):
    if isinstance(other, np.ma.masked_array):
        self.data.%(method)s (other.data)
        self.mask = np.ma.mask_or(self.mask, other.mask)
    else:
        self.data.%(method)s (other)
    return self
""" % {'method':method})
    
    def integrate(self, nu, tf, dt=0.001, theta=0.001, ism=True, m=None, frozen=None):
        num_pops = self.num_pops
        
        if tf < 0:
            raise ValueError("integration time must be positive")
        
        if tf == 0.0:
            return
        
        self.data[:] = integrate_het(self.data, nu, tf, dt=dt, 
                                     theta=theta, m=m, ism=ism,
                                     num_pops = num_pops, frozen=frozen)
    
