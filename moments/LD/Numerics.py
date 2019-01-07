import numpy as np

import copy
import itertools
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as spinv

### XXX want to silence sparseefficiencywarning - or even better, make it more efficient
### I've commented out the networkx construction for multipopulation modeling - will eventually get rid of it completely

from moments.LD import Matrices

import networkx as nx
import pickle
import itertools

### one pop numerics

# for a given order D^n
names = {}
lengths = {}
def moment_names_onepop(n):
    """
    n : order for which we want the moment name list
    """
    n = int(n)
    try:
        all_moments = names[n]
    except KeyError:
        all_moments = []
        if n%2: # odd
            order = 1
            last_moments = ['D^1','z']
            all_moments = last_moments + all_moments
        else: # even
            order = 0
            last_moments = ['1']
            all_moments = last_moments + all_moments
        while order < n:
            order += 2
            last_sigma = []
            for mom in last_moments:
                if '_s' not in mom:
                    if mom == '1':
                        new_mom = ['1_sp1','1_sq1']
                        last_sigma.append(new_mom[0])
                        last_sigma.append(new_mom[1])
                    else:
                        new_mom = mom+'_s1'
                        last_sigma.append(new_mom)
                elif mom[:3] == '1_s':
                    s_order = int(mom[-1])
                    new_mom = mom[:-1] + '{0}'.format(s_order+1)
                    last_sigma.append(new_mom)
                else:
                    s_order = int(mom.split('_s')[1])
                    new_mom = mom.split('_s')[0]+'_s{0}'.format(s_order+1)
                    last_sigma.append(new_mom)
            new_moments = []
            for ii in range(order+1):
                pi_order = int(ii/2)
                z_order = ii%2
                if pi_order > 0:
                    if z_order > 0:
                        appendage = '_pi^{0}_z'.format(pi_order)
                    else:
                        appendage = '_pi^{0}'.format(pi_order)
                else:
                    if z_order > 0:
                        appendage = '_z'
                    else:
                        appendage = ''
                if ii < order:
                    new_mom = 'D^{0}'.format(order-ii) + appendage
                else:
                    new_mom = appendage[1:]
                new_moments.append(new_mom)
            all_moments = new_moments + last_sigma + all_moments
            last_moments = new_moments + last_sigma
        names[n] = all_moments
        lengths[len(all_moments)] = n
    return all_moments

def moment_names_multipop(num_pops):
    """
    returns a list of moment names when there are num_pops 
        number of populations present
    """
    all_moments = []
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            all_moments.append('DD_{0}_{1}'.format(i,j))
    for i in range(1,num_pops+1):
        for j in range(1,num_pops+1):
            for k in range(1,num_pops+1):
                all_moments.append('Dz_{0}_{1}_{2}'.format(i,j,k))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            for k in range(1,num_pops+1):
                for l in range(k,num_pops+1):
                    all_moments.append('zz_{0}_{1}_{2}_{3}'.format(i,j,k,l))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            all_moments.append('zp_{0}_{1}'.format(i,j))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            all_moments.append('zq_{0}_{1}'.format(i,j))
    return all_moments+['1']

def moment_names_pi(num_pops):
    """
    returns a list of moment names in the pi basis (order 2) 
        when there are mulitple populations
    """
    all_moments = []
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            all_moments.append('DD_{0}_{1}'.format(i,j))
    for i in range(1,num_pops+1):
        for j in range(1,num_pops+1):
            for k in range(1,num_pops+1):
                all_moments.append('Dz_{0}_{1}_{2}'.format(i,j,k))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            for k in range(1,num_pops+1):
                for l in range(k,num_pops+1):
                    all_moments.append('pi2_{0}_{1}_{2}_{3}'.format(i,j,k,l))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            all_moments.append('pi1p_{0}_{1}'.format(i,j))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            all_moments.append('pi1q_{0}_{1}'.format(i,j))
    return all_moments+['1']

mom_map = {}
mom_map['pi'] = {}
mom_map['z'] = {}
def map_moment(mom, basis):
    try:
        return mom_map[basis][mom]
    except KeyError:
        if mom.split('_')[0] == 'DD':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'DD_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'zz':
            popsp = sorted([int(p) for p in mom.split('_')[1:3]])
            popsq = sorted([int(p) for p in mom.split('_')[3:]])
            mom_out = 'zz_'+'_'.join([str(p) for p in popsp])+'_'+'_'.join([str(p) for p in popsq])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'zp':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'zp_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'zq':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'zq_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'pi2':
            popsp = sorted([int(p) for p in mom.split('_')[1:3]])
            popsq = sorted([int(p) for p in mom.split('_')[3:]])
            mom_out = 'pi2_'+'_'.join([str(p) for p in popsp])+'_'+'_'.join([str(p) for p in popsq])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'pi1p':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'pi1p_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        elif mom.split('_')[0] == 'pi1q':
            pops = sorted([int(p) for p in mom.split('_')[1:]])
            mom_out = 'pi1q_'+'_'.join([str(p) for p in pops])
            mom_map[mom] = mom_out
        else:
            mom_out = mom
        mom_map[basis][mom] = mom_out
        return mom_map[basis][mom] 


"""
Transition matrices for pi basis integration
"""

def drift(num_pops, order, nus=None):
    if num_pops != 1 and order != 2:
        raise ValueError("not okay")
    
    if num_pops == 1:
        row = []
        col = []
        data = []
        corner = 0
        while order >= 0:
            row_current = Matrices.drift_dict[order]['row']
            new_row = [x+corner for x in row_current]
            col_current = Matrices.drift_dict[order]['col']
            new_col = [x+corner for x in col_current]
            data_current = Matrices.drift_dict[order]['data']
            size = Matrices.drift_dict[order]['size']
            row.extend(new_row)
            col.extend(new_col)
            data.extend(data_current)
            corner += size[0]
            order -= 2
        return csc_matrix((data,(row,col)),shape=(corner,corner))
    else:
        if num_pops == 2:
            return Matrices.drift_two_pop_pi(nus)    
        elif num_pops == 3:
            return Matrices.drift_three_pop_pi(nus)
        elif num_pops == 4:
            return Matrices.drift_four_pop_pi(nus)
        elif num_pops == 5:
            return Matrices.drift_five_pop_pi(nus)
        else:
            print("not done")
            return

def mutation(num_pops, order, theta, ism):
    """
    theta : list of two mutation rates [theta_left, theta_right]
    ism : True or False (ism=False for reversible mutation model)
    """
    if num_pops == 1:
        if ism == False:
            theta = theta[0]
            corner = 0
            row = []
            col = []
            data = []
            while order >= 0:
                row_current = Matrices.mut_dict[order]['row']
                new_row = [x+corner for x in row_current]
                col_current = Matrices.mut_dict[order]['col']
                new_col = [x+corner for x in col_current]
                data_current = Matrices.mut_dict[order]['data']
                size = Matrices.mut_dict[order]['size']
                row.extend(new_row)
                col.extend(new_col)
                data.extend(data_current)
                corner += size[0]
                order -= 2
            return theta * csc_matrix((data,(row,col)),shape=(corner,corner))
        elif ism == True:
            theta1,theta2 = theta
            # ISM model only built for even orders
            mom_list = moment_names_onepop(order)
            size = len(mom_list)
            M = np.zeros((size,size))
            # [pi s_{i}]_{t+1} = [pi s_{i}]_{t} + theta/2 [s_{i+1}]_{t}
            M[mom_list.index('1_sp1'), mom_list.index('1')] = 1./2 * theta1
            M[mom_list.index('1_sq1'), mom_list.index('1')] = 1./2 * theta2
            M[mom_list.index('pi^1'), mom_list.index('1_sp1')] = 1./2 * theta2
            M[mom_list.index('pi^1'), mom_list.index('1_sq1')] = 1./2 * theta1
            for ii in range(1,int(order/2)):
                M[mom_list.index('pi^1_s{0}'.format(ii)), mom_list.index('1_sp{0}'.format(ii+1))] = 1./2 * theta2
                M[mom_list.index('pi^1_s{0}'.format(ii)), mom_list.index('1_sq{0}'.format(ii+1))] = 1./2 * theta1
        
        return csc_matrix(M)
    else:
        if ism == False:
            theta = theta[0]
            if num_pops == 2:
                return Matrices.mutat_two_pop_pi(theta)
            if num_pops == 3:
                return Matrices.mutat_three_pop_pi(theta)
            if num_pops == 4:
                return Matrices.mutat_four_pop_pi(theta)
            else:
                print("not done")
                return
        else:
            theta1,theta2 = theta
            mom_list = moment_names_pi(num_pops)

            row = []
            col = []
            data = []
            for mom in mom_list:
                if mom.split('_')[0] == 'pi1p':
                    row.append(mom_list.index(mom))
                    col.append(mom_list.index('1'))
                    data.append(theta1/2.)
                elif mom.split('_')[0] == 'pi1q':
                    row.append(mom_list.index(mom))
                    col.append(mom_list.index('1'))
                    data.append(theta2/2.)
                elif mom.split('_')[0] == 'pi2':
                    mom1 = 'pi1p_'+ mom.split('_')[1] +'_' + mom.split('_')[2]
                    mom2 = 'pi1q_'+ mom.split('_')[3] +'_' + mom.split('_')[4]
                    row.append(mom_list.index(mom))
                    col.append(mom_list.index(mom1))
                    data.append(theta2/2.)
                    row.append(mom_list.index(mom))
                    col.append(mom_list.index(mom2))
                    data.append(theta1/2.)
            return csc_matrix((data,(row,col)),shape=((len(mom_list),len(mom_list))))


def recombination(num_pops, order, rho=0.0):
    row = []
    data = []
    if num_pops == 1 and order != 2:
        moms = moment_names_onepop(order)
        for ii,moment in enumerate(moms):
            if 'D' in moment:
                D_order = int(moment.split('_')[0].split('^')[1])
                row.append(ii)
                data.append(-D_order/2.)
            else:
                continue
        size = (len(moms),len(moms))
        return csc_matrix((data,(row,row)),shape=size)
    else:
        return recombination_multipop(rho,num_pops)

def migration(num_pops, ms, ism=True):
    if num_pops == 2:
        return Matrices.migra_two_pop_pi(ms)    
    elif num_pops == 3:
        return Matrices.migra_three_pop_pi(ms)
    elif num_pops == 4:
        return Matrices.migra_four_pop_pi(ms)
    elif num_pops == 5:
        return Matrices.migra_five_pop_pi(ms)
    else:
        print("not done")
        return

"""
Integration routine for pi basis
"""

def integrate(y, nu, T, dt=0.001, rho=0.0, theta=0.0008, m=[], order=None, num_pops=1, ism=True):
    """
    
    """
    if num_pops == 1:
        if callable(nu) == False:
            nu = np.float(nu[0])
        
        if hasattr(theta, '__len__') == False:
            theta = [np.float(theta), np.float(theta)]
        
        rho = np.float(rho)
        
        if order is None:
            try:
                order = lengths[len(y)]
            except KeyError:
                raise KeyError("specify order or get moment names")
        
        moms = moment_names_onepop(order)
        if len(y) != len(moms):
            raise ValueError("there is a vector size mismatch")
        
        # found that using numpy linalg inverse was faster than scipy sparse solver 
        # for this size system, though admittedly not that great...
#        D = drift(1, order)
#        M = mutation(1, order, theta, ism)
#        R = recombination(num_pops, order, rho=rho)
        
        D = drift(1, order).toarray()
        M = mutation(1, order, theta, ism).toarray()
        R = recombination(num_pops, order, rho=rho).toarray()
        EYE = np.eye(D.shape[0])
        
        N = 1.0
        
        elapsed_t = 0
        while elapsed_t < T:
            # ensure that final integration time does not exceed T
            if elapsed_t + dt > T:
                dt = T-elapsed_t
            
            # if nu is a function, set N to nu(t+dt/2)
            if callable(nu):
                N = np.float(nu(elapsed_t + dt/2.)[0])
            else:
                N = nu
            
            if elapsed_t == 0 or dt != dt_old or N != N_old:
                A = D/N + M + R
                Afd = EYE + dt/2.*A
                Abd = np.linalg.inv(EYE - dt/2.*A)
#                Afd = identity(A.shape[0], format='csc') + dt/2.*A
#                Abd = factorized(identity(A.shape[0], format='csc') - dt/2.*A)
            
            y = Abd.dot(Afd.dot(y))
#            y = Abd(Afd.dot(y))
            elapsed_t += dt
            dt_old = dt
            N_old = N
    elif num_pops > 1:
        moms = moment_names_pi(num_pops)
        if len(moms) != len(y):
            raise ValueError("num_pops must be set to correct number of populations")
        
        if num_pops > 1 and m is not None:
            if m == []:
                ms = num_pops*(num_pops-1)*[0]
            else:
                ms = []
                for ii in range(num_pops):
                    for jj in range(ii+1,num_pops):
                        # note that in Matrices, we've reversed the meaning of m_ij (easier to fix here)
                        ms.append(m[jj][ii])
                        ms.append(m[ii][jj])
            ms = [float(mval) for mval in ms]
            M = migration(num_pops, ms)
        
        R = recombination(num_pops, order, rho=rho)
            
        if hasattr(theta, '__len__') == False:
            theta = [np.float(theta), np.float(theta)]
        
        U = mutation(num_pops, 2, theta, ism)
    
        if callable(nu):
            nus = nu(0)
        else:
            nus = [float(nu_pop) for nu_pop in nu]
        
        dt_last = dt
        nus_last = nus
        elapsed_T = 0
        while elapsed_T < T:
            
            if elapsed_T + dt > T:
                dt = T-elapsed_T
            
            if callable(nu):
                nus = nu(elapsed_T+dt/2.)
            
            if dt != dt_last or nus != nus_last or elapsed_T == 0:
                D = drift(num_pops, order, nus=nus)
                if num_pops > 1 and m is not None:
                    Ab = D+M+R+U
                else:
                    Ab = D+R+U
                Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
                Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
            
            y = Ab2(Ab1.dot(y))
            elapsed_T += dt
            dt_last = copy.copy(dt)
            nus_last = copy.copy(nus)

    return y

def equilibrium(rho=0.0, theta=0.0008, ism=True, order=2):
    rho = np.float(rho)
    if hasattr(theta, '__len__') == False:
        theta = [np.float(theta), np.float(theta)]
    D = drift(1, order)
    R = recombination(1, order, rho=rho)
    M = mutation(1, order, theta, ism)
    A = (D + M + R).toarray()
    B = A[:-1,-1]
    A = A[:-1,:-1]
    y0 = np.linalg.inv(A).dot(-B)
    return np.concatenate(( y0, [1] ))



"""
Matrices for multipopulation integration
For numbers of populations 1-4 we have cached the csc matrices to build quickly
For larger number of pops (>4) we build them here
"""

def drift_multipop(nus,num_pops):
    nus = [float(nu) for nu in nus]
    if num_pops == 1:
        return Matrices.drift_one_pop(nus[0])
    elif num_pops == 2:
        return Matrices.drift_two_pop(nus)    
    elif num_pops == 3:
        return Matrices.drift_three_pop(nus)
    elif num_pops == 4:
        return Matrices.drift_four_pop(nus)
    else:
        mom_list = moment_names_multipop(num_pops)
        row = []
        col = []
        data = []
        
        for mom in mom_list:
            this_ind = mom_list.index(mom)
            mom2s, vals = Matrices.drift_multipop_terms(mom,nus)
            for mom2,val in zip(mom2s,vals):
                row.append(this_ind)
                col.append(mom_list.index(mom2))
                data.append(val)
        return csc_matrix((data,(row,col)),shape=((len(mom_list),len(mom_list))))

def migration_multipop(ms,num_pops):
    """
    ms has form [m12, m21, m13, m31, m14, m41, ..., m23, m32, m24, m42, ...]
    """
    if num_pops == 2:
        return Matrices.migra_two_pop(ms)    
    elif num_pops == 3:
        return Matrices.migra_three_pop(ms)
    elif num_pops == 4:
        return Matrices.migra_four_pop(ms)
    else:
        if num_pops*(num_pops-1) != len(ms):
            raise ValueError("mismatch between number of populations and input number of migration rates")
        mom_list = moment_names_multipop(num_pops)
        M = np.zeros((len(mom_list),len(mom_list)))
        
        for mom in mom_list:
            this_ind = mom_list.index(mom)
            mom2s, vals = Matrices.migration_multipop_terms(mom, ms, num_pops)
            # mom2s are not always in the format of mom_list, so we map them to the correct moment name
            for mom2,val in zip(mom2s,vals):
                M[this_ind, mom_list.index(map_moment(mom2, 'z'))] = val
        return csc_matrix(M)

def recombination_multipop(rho,num_pops):
    if num_pops == 1:
        return Matrices.recom_one_pop(rho)
    elif num_pops == 2:
        return Matrices.recom_two_pop(rho)    
    elif num_pops == 3:
        return Matrices.recom_three_pop(rho)
    elif num_pops == 4:
        return Matrices.recom_four_pop(rho)
    else:
        mom_list = moment_names_multipop(num_pops)
        row = []
        col = []
        data = []
        for ii,mom in enumerate(mom_list):
            if mom.split('_')[0] == 'DD':
                row.append(ii)
                col.append(ii)
                data.append(-rho)
            elif mom.split('_')[0] == 'Dz':
                row.append(ii)
                col.append(ii)
                data.append(-rho/2.)
        return csc_matrix((data,(row,col)),shape=((len(mom_list),len(mom_list))))

def mutation_multipop(mu,num_pops,ism=True):
    if ism == False:
        mu = mu[0] # mu is a list of length two for theta_left and theta_right, but we have equal mutation rates in the reversible model
        if num_pops == 1:
            return Matrices.mutat_one_pop(mu)
        elif num_pops == 2:
            return Matrices.mutat_two_pop(mu)    
        elif num_pops == 3:
            return Matrices.mutat_three_pop(mu)
        elif num_pops == 4:
            return Matrices.mutat_four_pop(mu)
        else:
            mom_list = moment_names_multipop(num_pops)
            row = []
            col = []
            data = []
            for ii,mom in enumerate(mom_list):
                if mom.split('_')[0] in ['DD','Dz','zz']:
                    row.append(ii)
                    col.append(ii)
                    data.append(-4.*mu)
                elif mom.split('_')[0] in ['zp','zq']:
                    row.append(ii)
                    col.append(ii)
                    data.append(-2.*mu)
            return csc_matrix((data,(row,col)),shape=((len(mom_list),len(mom_list))))
    elif ism == True:
        mu1,mu2 = mu
        mom_list = moment_names_multipop(num_pops)
        #M = np.zeros((len(mom_list), len(mom_list)))
        row = []
        col = []
        data = []
        for mom in mom_list:
            if mom.split('_')[0] == 'zp':
                row.append(mom_list.index(mom))
                col.append(mom_list.index('1'))
                data.append(-2.*mu1)
            elif mom.split('_')[0] == 'zq':
                row.append(mom_list.index(mom))
                col.append(mom_list.index('1'))
                data.append(-2.*mu2)
            elif mom.split('_')[0] == 'zz':
                mom1 = 'zp_'+ mom.split('_')[1] +'_' + mom.split('_')[2]
                mom2 = 'zq_'+ mom.split('_')[3] +'_' + mom.split('_')[4]
                row.append(mom_list.index(mom))
                col.append(mom_list.index(mom1))
                data.append(-2.*mu2)
                row.append(mom_list.index(mom))
                col.append(mom_list.index(mom2))
                data.append(-2.*mu1)
        return csc_matrix((data,(row,col)),shape=((len(mom_list),len(mom_list))))

"""
Integration for multiple populations in style of moments
"""

def integrate_multipop(y, nu, T, num_pops=1, rho=0.0, theta=0.0008, dt=0.001, m=[], ism=True):
    """
    Integration function for multipopulation statistics
    y: LDstats object with y.data, y.num_pops, y.order (=2 for multipop models)
    nu: list of relative population sizes, with length equal to y.num_pops, can be function in time
    T: time to integrate
    rho: scale recombination rate 4Nr
    theta: scale mutation rate 4Nmu
    dt: time step to use
    m: migration matrix with [[0, m12, m13, ...],[m21, 0, m23, ...],...]
    
    Note that in the multipopulation basis, only the reversible mutation model is possible
    """
    moms = moment_names_multipop(num_pops)
    if len(moms) != len(y):
        raise ValueError("num_pops must be set to correct number of populations")
    
    if num_pops > 1 and m is not None:
        if m == []:
            ms = num_pops*(num_pops-1)*[0]
        else:
            ms = []
            for ii in range(num_pops):
                for jj in range(ii+1,num_pops):
                    # note that in Matrices, we've reversed the meaning of m_ij (easier to fix here)
                    ms.append(m[jj][ii])
                    ms.append(m[ii][jj])
        ms = [float(mval) for mval in ms]
        M = migration_multipop(ms,num_pops)
    
    R = recombination_multipop(rho,num_pops)
    
    if hasattr(theta, '__len__') == False:
        theta = [np.float(theta), np.float(theta)]
    
    U = mutation_multipop(theta,num_pops,ism=ism)
    
    if callable(nu):
        nus = nu(0)
    else:
        nus = [float(nu_pop) for nu_pop in nu]
    
    #D = drift_multipop(nus,num_pops)
    
    dt_last = dt
    nus_last = nus
    elapsed_T = 0
    # improve with t_elapsed, below checking if pop sizes changed
    while elapsed_T < T:
        
        if elapsed_T + dt > T:
            dt = T-elapsed_T
        
        if callable(nu):
            nus = nu(elapsed_T+dt/2.)
        
        if dt != dt_last or nus != nus_last or elapsed_T == 0:
            D = drift_multipop(nus,num_pops)
            if num_pops > 1 and m is not None:
                Ab = D+M+R+U
            else:
                Ab = D+R+U
            Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
            Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
        
        y = Ab2(Ab1.dot(y))
        elapsed_T += dt
        dt_last = copy.copy(dt)
        nus_last = copy.copy(nus)
    
    return y

def equilibrium_multipop(rho, theta, dt=0.01, ism=True):
    D = drift_multipop([1.],1)
    R = recombination_multipop(rho,1)
    U = mutation_multipop(theta,1,ism=ism)
    
    Ab = (D+R+U).toarray()
    
    b = Ab[:-1,-1]
    A = Ab[:-1,:-1]
    y0 = np.linalg.inv(A).dot(-b)
    return np.concatenate(( y0, [1] ))


"""
Manipulations such as admixture and merging
"""

def merge_2pop(y, f):
    """
    admixture event between two populations, but we don't keep initial pops
    takes a two pop LDstats object, returns a one pop object
    """
    y_new = np.ones(6)
    mns = moment_names_multipop(2)
    DD = f**2 * y[mns.index('DD_1_1')] + 2*(f*(1-f)) * y[mns.index('DD_1_2')] \
          + (1-f)**2 * y[mns.index('DD_2_2')] \
          + 0.5*f**2*(1-f) * y[mns.index('Dz_1_1_1')] - 0.5*f**2*(1-f) * y[mns.index('Dz_1_1_2')] \
          - 0.5*f**2*(1-f) * y[mns.index('Dz_1_2_1')] + 0.5*f**2*(1-f) * y[mns.index('Dz_1_2_2')] \
          + 0.5*f*(1-f)**2 * y[mns.index('Dz_2_1_1')] - 0.5*f*(1-f)**2 * y[mns.index('Dz_2_1_2')] \
          - 0.5*f*(1-f)**2 * y[mns.index('Dz_2_2_1')] + 0.5*f*(1-f)**2 * y[mns.index('Dz_2_2_2')] \
          + 1./16*f**2*(1-f)**2 * y[mns.index('zz_1_1_1_1')] \
          - 1./8*f**2*(1-f)**2 * y[mns.index('zz_1_1_1_2')] + 1./16*f**2*(1-f)**2 * y[mns.index('zz_1_1_2_2')] \
          - 1./8*f**2*(1-f)**2 * y[mns.index('zz_1_2_1_1')] + 1./4*f**2*(1-f)**2 * y[mns.index('zz_1_2_1_2')] \
          - 1./8*f**2*(1-f)**2 * y[mns.index('zz_1_2_2_2')] + 1./16*f**2*(1-f)**2 * y[mns.index('zz_2_2_1_1')] \
          - 1./8*f**2*(1-f)**2 * y[mns.index('zz_2_2_1_2')] + 1./16*f**2*(1-f)**2 * y[mns.index('zz_2_2_2_2')]
    Dz = f**3 * y[mns.index('Dz_1_1_1')] + f**2*(1-f) * y[mns.index('Dz_1_1_2')] \
          + f**2*(1-f) * y[mns.index('Dz_1_2_1')] + f*(1-f)**2 * y[mns.index('Dz_1_2_2')] \
          + f**2*(1-f) * y[mns.index('Dz_2_1_1')] + f*(1-f)**2 * y[mns.index('Dz_2_1_2')] \
          + f*(1-f)**2 * y[mns.index('Dz_2_2_1')] + (1-f)**3 * y[mns.index('Dz_2_2_2')] \
          + 1./4*f**3*(1-f) * y[mns.index('zz_1_1_1_1')] + 1./4*f**2*(1-f)*(1-2*f) * y[mns.index('zz_1_1_1_2')] \
          - 1./4*f**2*(1-f)**2 * y[mns.index('zz_1_1_2_2')] \
          + 1./4*f**2*(1-f)*(1-2*f) * y[mns.index('zz_1_2_1_1')] + 1./4*f*(1-f)*(1-2*f)**2 * y[mns.index('zz_1_2_1_2')] \
          - 1./4*f*(1-f)**2*(1-2*f) * y[mns.index('zz_1_2_2_2')] \
          - 1./4*f**2*(1-f)**2 * y[mns.index('zz_2_2_1_1')] - 1./4*f*(1-f)**2*(1-2*f) * y[mns.index('zz_2_2_1_2')] \
          + 1./4*f*(1-f)**3 * y[mns.index('zz_2_2_2_2')]
    zz = f**4 * y[mns.index('zz_1_1_1_1')] + 2*f**3*(1-f) * y[mns.index('zz_1_1_1_2')] \
          + f**2*(1-f)**2 * y[mns.index('zz_1_1_2_2')] \
          + 2*f**3*(1-f) * y[mns.index('zz_1_2_1_1')] + 4*f**2*(1-f)**2 * y[mns.index('zz_1_2_1_2')] \
          + 2*f*(1-f)**3 * y[mns.index('zz_1_2_2_2')] \
          + f**2*(1-f)**2 * y[mns.index('zz_2_2_1_1')] + 2*f*(1-f)**3 * y[mns.index('zz_2_2_1_2')] \
          + (1-f)**4 * y[mns.index('zz_2_2_2_2')]
    zp = f**2 * y[mns.index('zp_1_1')] + 2*f*(1-f) * y[mns.index('zp_1_2')] + (1-f)**2 * y[mns.index('zp_2_2')]
    zq = f**2 * y[mns.index('zq_1_1')] + 2*f*(1-f) * y[mns.index('zq_1_2')] + (1-f)**2 * y[mns.index('zq_2_2')]
    y_new[0] = DD
    y_new[1] = Dz
    y_new[2] = zz
    y_new[3] = zp
    y_new[4] = zq
    return y_new

def admix_npops(y, n_pops, pop1, pop2, f):
    """
    New population is appended
    f from pop1, 1-f from pop2
    """
    moms_from = moment_names_multipop(n_pops)
    moms_to = moment_names_multipop(n_pops+1)
    A = Matrices.admix_npops(n_pops, pop1, pop2, f, moms_from, moms_to)
    y_new = A.dot(y)
    return y_new
