import numpy as np

import copy

from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as spinv

### XXX want to silence sparseefficiencywarning - or even better, make it more efficient

from moments.LD import Matrices

import networkx as nx
import cPickle as pickle
import itertools

### one pop numerics

# for a given order D^n
names = {}
lengths = {}
def moment_names_onepop(n):
    n = int(n)
    try:
        moments = names[n]
    except KeyError:
        moments = []
        if n%2: # odd
            order = 1
            last_moments = ['D^1','z']
            moments = last_moments + moments
        else: # even
            order = 0
            last_moments = ['1']
            moments = last_moments + moments
        while order < n:
            order += 2
            last_sigma = []
            for mom in last_moments:
                if '_s' not in mom:
                    new_mom = mom+'_s1'
                else:
                    s_order = int(mom.split('_s')[1])
                    new_mom = mom.split('_s')[0]+'_s{0}'.format(s_order+1)
                last_sigma.append(new_mom)
            new_moments = []
            for ii in range(order+1):
                pi_order = ii/2
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
            moments = new_moments + last_sigma + moments
            last_moments = new_moments + last_sigma
        names[n] = moments
        lengths[len(moments)] = n
    return moments

def moment_names_multipop(n):
    return moment_list(n)

### single population transition matrices
def drift(n):
    order = int(n)
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

def mutation(n, ism):
    order = int(n)
    row = []
    col = []
    data = []
    if ism == False:
        corner = 0
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
        return csc_matrix((data,(row,col)),shape=(corner,corner))
    elif ism == True:
        # ISM model only built for even orders
        names = moment_names_onepop(n)
        size = len(names)
        M = np.zeros((size,size))
        # [pi s_{i}]_{t+1} = [pi s_{i}]_{t} + theta/2 [s_{i+1}]_{t}
        M[names.index('1_s1'), names.index('1')] = 1.0
        M[names.index('pi^1'), names.index('1_s1')] = 1./2
        for ii in range(1,n/2):
            M[names.index('pi^1_s{0}'.format(ii)), names.index('1_s{0}'.format(ii+1))] = 1./2
        
        return csc_matrix(M)

def recombination(n):
    row = []
    data = []
    moms = moment_names_onepop(n)
    for ii,moment in zip(range(len(moms)),moms):
        if 'D' in moment:
            D_order = int(moment.split('_')[0].split('^')[1])
            row.append(ii)
            data.append(-D_order/2.)
        else:
            continue
    size = (len(moms),len(moms))
    return csc_matrix((data,(row,row)),shape=size)

def integrate(y, T, rho=0.0, nu=1.0, theta=0.0008, order=None, dt=0.001, ism=False):
    if callable(nu) == False:
        nu = np.float(nu)
    theta = np.float(theta)
    rho = np.float(rho)
    if order is None:
        try:
            order = lengths[len(y)]
        except KeyError:
            raise KeyError("specify order or get moment names")
    
    moms = moment_names_onepop(order)
    if len(y) != len(moms):
        raise ValueError("there is a vector size mismatch")

    
#    D = drift(order)
#    M = mutation(order, ism)
#    R = recombination(order)
        
    D = drift(order).toarray()
    M = mutation(order, ism).toarray()
    R = recombination(order).toarray()
    EYE = np.eye(D.shape[0])
    
    N = 1.0
    
    elapsed_t = 0
    while elapsed_t < T:
        # ensure that final integration time does not exceed T
        if elapsed_t + dt > T:
            dt = T-elapsed_t
        
        # if nu is a function, set N to nu(t+dt/2)
        if callable(nu):
            N = np.float(nu(elapsed_t + dt/2.))
        else:
            N = nu
        
        if elapsed_t == 0 or dt != dt_old or N != N_old:
            A = D/N + M*theta + R*rho
#            Afd = identity(A.shape[0], format='csc') + dt/2.*A
            Afd = EYE + dt/2.*A
#            Abd = factorized(identity(A.shape[0], format='csc') - dt/2.*A)
            Abd = np.linalg.inv(EYE - dt/2.*A)
        
#        y = Abd(Afd.dot(y))
        y = Abd.dot(Afd.dot(y))
        elapsed_t += dt
        dt_old = dt
        N_old = N
    
    return y

def equilibrium(order=2, rho=0.0, theta=0.0008, ism=False):
    D = drift(order)
    R = recombination(order)
    M = mutation(order, ism)
    A = D + M*theta + R*rho
    B = A[:-1,-1]
    A = A[:-1,:-1]
    return np.concatenate(( np.ndarray.flatten(np.array(-factorized(A)(B.todense()))), np.array([1]) ))


### multi pop numerics

"""
These are for dealing with networkx defined demography trees (see example)
Initially I build models using networkx graphs and hacked together ways to deal
with population splits and demography. Admittedly, it's not pretty and as
usually, the comments are shit.
Much farther below is integration in style of moments
"""

def get_leaves(demo):
    """
    get the leaves (contemporary populations) from the networkx tree
    """
    leaves = []
    for node in demo.nodes():
        if demo.successors(node) == []:
            leaves.append(node)
    return sorted(leaves)

def get_interiors(demo):
    """
    get the interior (ancestral) populations
    """
    node_list = demo.nodes()
    leaves = get_leaves(demo)
    for leaf in leaves:
        node_list.remove(leaf)
    return sorted(node_list)

def get_root(demo):
    """
    get the common ancestor population from the networkx tree
    """
    leaves = get_leaves(demo)
    possible_roots = []
    for set in itertools.permutations(leaves, len(leaves)):
        possible_roots.append('-'.join(set))
    for pr in possible_roots:
        if pr in demo.node.keys():
            root = pr
    return root
    
def add_times(demo, tree_times):
    """
    demo: the population tree with nodes (populations) and edges (pop relations)
    tree_times: dictionary with {internal_node_name: time_of_split}, where
                times are measured from the beginning of the simulation (from equilibrium
                of the root node)
                also contains {'F': T}, the time to end the simulation (present time)
    this adds the times to the demo object, which is needed to integrate moments
        along the tree
    """
    leaves = get_leaves(demo)
    interiors = get_interiors(demo)
    for node in interiors:
        if node == get_root(demo):
            demo.node[node]['Ttop'] = 0.0
            demo.node[node]['Tbot'] = tree_times[node]
        else:
            parent = demo.predecessors(node)[0]
            demo.node[node]['Ttop'] = tree_times[parent]
            demo.node[node]['Tbot'] = tree_times[node]
    for node in leaves:
        parent = demo.predecessors(node)[0]
        demo.node[node]['Ttop'] = tree_times[parent]
        demo.node[node]['Tbot'] = tree_times['F']
    return demo

def get_event_times(demo):
    """
    returns sorted list of split/epoch times
    """
    times = [0.0]
    leaves = get_leaves(demo)
    interiors = get_interiors(demo)
    # add split times
    for node in interiors:
        times.append(demo.node[node]['Tbot'])
    # add final time
    times.append(demo.node[leaves[0]]['Tbot'])
    times = list(set(times))
    times = sorted(times)
    return times


def get_current_pops(demo, T):
    """
    given demography demo with tree_times already added, returns which populations
    are present at time T
    """
    if T == demo.node[get_leaves(demo)[0]]['Tbot']:
        return get_leaves(demo)
    else:
        pops = []
        for node in get_interiors(demo):
            if T >= demo.node[node]['Ttop'] and T < demo.node[node]['Tbot']:
                pops.append(node)
        for node in get_leaves(demo):
            if T >= demo.node[node]['Ttop'] and T < demo.node[node]['Tbot']:
                pops.append(node)
        return pops

def get_migration_rates(demo,T,ordered_pops):
    npops = len(ordered_pops)
    ms = np.zeros((npops)*(npops-1))
    m_count = 0
    for i in range(npops):
        pop1 = ordered_pops[i]
        for j in range(i+1,npops):
            pop2 = ordered_pops[j]
            if pop2 in demo.node[pop1]['mig'].keys():
                ms[m_count] = demo.node[pop1]['mig'][pop2]
            m_count +=1 
            if pop1 in demo.node[pop2]['mig'].keys():
                ms[m_count] = demo.node[pop2]['mig'][pop1]
            m_count += 1
    return ms

def pop_size_from_exp(T,Ttop,Tbot,Ntop,Nbot):
    return Ntop * np.exp(np.log(float(Nbot)/float(Ntop)) * (T-Ttop)/(Tbot-Ttop))

def get_pop_sizes(demo, T, pops):
    nus = []
    for pop in pops:
        if demo.node[pop]['model'] == 'constant':
            nus.append(demo.node[pop]['nu'])
        elif demo.node[pop]['model'] == 'exponential':
            nus.append(pop_size_from_exp(T,demo.node[pop]['Ttop'],demo.node[pop]['Tbot'],demo.node[pop]['nu_top'],demo.node[pop]['nu_bot']))
        else:
            raise "haven't implemented non-constant/exponential demography"
    return nus

def demography(demo, rho=0.0, theta=1e-4, dt=0.0001):
    event_times = get_event_times(demo)
    y, ordered_pops = root_equilibrium_nx(demo, rho, theta, dt)
    for period in zip(event_times[:-1],event_times[1:]):
        y, ordered_pops = seed_after_split(demo, period[0], y, ordered_pops)
        y = integrate_multipop_nx(demo, period[0], period[1], rho, theta, y, ordered_pops, dt)
    return y[:-1], ordered_pops

"""
Multipopulation moment names and seeding statistics after a population split
"""

def moment_list(num_pops):
    """
    returns a list of moment names when there are num_pops number of populations present
    """
    ll = []
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            ll.append('DD_{0}_{1}'.format(i,j))
    for i in range(1,num_pops+1):
        for j in range(1,num_pops+1):
            for k in range(1,num_pops+1):
                ll.append('Dz_{0}_{1}_{2}'.format(i,j,k))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            for k in range(1,num_pops+1):
                for l in range(k,num_pops+1):
                    ll.append('zz_{0}_{1}_{2}_{3}'.format(i,j,k,l))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            ll.append('zp_{0}_{1}'.format(i,j))
    for i in range(1,num_pops+1):
        for j in range(i,num_pops+1):
            ll.append('zq_{0}_{1}'.format(i,j))
    return ll


def seed_after_split(demo, T, y_last, ordered_pops):
    # y_last: vector of computed moments before split
    # ordered_pops_last: list of ordered pops from previous demo
    new_ordered_pops = []
    current_pops = get_current_pops(demo,T)
    for prev_pop in ordered_pops:
        if prev_pop in current_pops:
            new_ordered_pops.append(prev_pop)
        else:
            # previous pop split, e.g. 'A-B-C-D-E-F-G' -> 'A-B','C-D-E-F-G'
            num_dashes = prev_pop.count('-')
            splits = prev_pop.split('-')
            possible_splits = [('-'.join(splits[:i]),'-'.join(splits[i:])) for i in range(1,num_dashes+1)]
            for pos_split in possible_splits:
                if pos_split[0] in current_pops and pos_split[1] in current_pops:
                    new_ordered_pops.append(pos_split[0])
                    new_ordered_pops.append(pos_split[1])
    
    ml_last = moment_list(len(ordered_pops))
    ml = moment_list(len(new_ordered_pops))
    y = np.zeros(len(ml))
    
    for i,moment in zip(range(len(ml)),ml):
        # get from parent
        if moment.split('_')[0] == 'DD':
            pop1 = new_ordered_pops[int(moment.split('_')[1])-1]
            pop2 = new_ordered_pops[int(moment.split('_')[2])-1]
            if pop1 not in ordered_pops:
                parent1 = demo.predecessors(pop1)[0]
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    y[i] = y_last[ml_last.index('DD_{0}_{1}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1))]
                else:
                    y[i] = y_last[ml_last.index('DD_{0}_{1}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1))]                        
            else:
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    y[i] = y_last[ml_last.index('DD_{0}_{1}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1))]
                else:
                    y[i] = y_last[ml_last.index('DD_{0}_{1}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1))]                        
        
        elif moment.split('_')[0] == 'Dz':
            pop1 = new_ordered_pops[int(moment.split('_')[1])-1]
            pop2 = new_ordered_pops[int(moment.split('_')[2])-1]
            pop3 = new_ordered_pops[int(moment.split('_')[3])-1]
            if pop1 not in ordered_pops:
                parent1 = demo.predecessors(pop1)[0]
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(parent3)+1))]
                    else:
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(pop3)+1))]
                else:
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1))]
                    else:
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1))]
            else:
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(parent3)+1))]
                    else:
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(pop3)+1))]
                else:
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1))]
                    else:
                        y[i] = y_last[ml_last.index('Dz_{0}_{1}_{2}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(pop3)+1))]
            
        elif moment.split('_')[0] == 'zz':
            pop1 = new_ordered_pops[int(moment.split('_')[1])-1]
            pop2 = new_ordered_pops[int(moment.split('_')[2])-1]
            pop3 = new_ordered_pops[int(moment.split('_')[3])-1]
            pop4 = new_ordered_pops[int(moment.split('_')[4])-1]
            if pop1 not in ordered_pops:
                parent1 = demo.predecessors(pop1)[0]
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(pop4)+1))]
                    else:
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(pop4)+1))]
                else:
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(pop4)+1))]
                    else:
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(pop4)+1))]
            else:
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(pop4)+1))]
                    else:
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(pop4)+1))]
                else:
                    if pop3 not in ordered_pops:
                        parent3 = demo.predecessors(pop3)[0]
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(parent3)+1,ordered_pops.index(pop4)+1))]
                    else:
                        if pop4 not in ordered_pops:
                            parent4 = demo.predecessors(pop4)[0]
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(parent4)+1))]
                        else:
                            y[i] = y_last[ml_last.index('zz_{0}_{1}_{2}_{3}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1,ordered_pops.index(pop3)+1,ordered_pops.index(pop4)+1))]

        elif moment.split('_')[0] == 'zp':
            pop1 = new_ordered_pops[int(moment.split('_')[1])-1]
            pop2 = new_ordered_pops[int(moment.split('_')[2])-1]
            if pop1 not in ordered_pops:
                parent1 = demo.predecessors(pop1)[0]
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    y[i] = y_last[ml_last.index('zp_{0}_{1}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1))]
                else:
                    y[i] = y_last[ml_last.index('zp_{0}_{1}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1))]
            else:
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    y[i] = y_last[ml_last.index('zp_{0}_{1}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1))]
                else:
                    y[i] = y_last[ml_last.index('zp_{0}_{1}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1))]
        
        elif moment.split('_')[0] == 'zq':
            pop1 = new_ordered_pops[int(moment.split('_')[1])-1]
            pop2 = new_ordered_pops[int(moment.split('_')[2])-1]
            if pop1 not in ordered_pops:
                parent1 = demo.predecessors(pop1)[0]
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    y[i] = y_last[ml_last.index('zq_{0}_{1}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(parent2)+1))]
                else:
                    y[i] = y_last[ml_last.index('zq_{0}_{1}'.format(ordered_pops.index(parent1)+1,ordered_pops.index(pop2)+1))]
            else:
                if pop2 not in ordered_pops:
                    parent2 = demo.predecessors(pop2)[0]
                    y[i] = y_last[ml_last.index('zq_{0}_{1}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(parent2)+1))]
                else:
                    y[i] = y_last[ml_last.index('zq_{0}_{1}'.format(ordered_pops.index(pop1)+1,ordered_pops.index(pop2)+1))]
        
        else:
            print "what the fuck did i do wrong..."
    
    return np.concatenate(( y, np.ones(1) )), new_ordered_pops

"""
Matrices for multipopulation integration
"""

def drift_multipop(nus,npops):
    if npops == 1:
        return Matrices.drift_one_pop(nus[0])
    elif npops == 2:
        return Matrices.drift_two_pop(nus)    
    elif npops == 3:
        return Matrices.drift_three_pop(nus)
    elif npops == 4:
        return Matrices.drift_four_pop(nus)
    else:
        raise "haven't put together {0}-pop drift matrix yet...".format(len(nus))

def migration_multipop(ms,npops):
    if npops == 2:
        return Matrices.migra_two_pop(ms)    
    elif npops == 3:
        return Matrices.migra_three_pop(ms)
    elif npops == 4:
        return Matrices.migra_four_pop(ms)
    else:
        raise "haven't put together {0}-pop migration matrix yet...".format(len(nus))

def recombination_multipop(rho,npops):
    if npops == 1:
        return Matrices.recom_one_pop(rho)
    elif npops == 2:
        return Matrices.recom_two_pop(rho)    
    elif npops == 3:
        return Matrices.recom_three_pop(rho)
    elif npops == 4:
        return Matrices.recom_four_pop(rho)
    else:
        raise "haven't put together {0}-pop recomb matrix yet...".format(len(nus))

def mutation_multipop(mu,npops):
    if npops == 1:
        return Matrices.mutat_one_pop(mu)
    elif npops == 2:
        return Matrices.mutat_two_pop(mu)    
    elif npops == 3:
        return Matrices.mutat_three_pop(mu)
    elif npops == 4:
        return Matrices.mutat_four_pop(mu)
    else:
        raise "haven't put together {0}-pop mutation matrix yet...".format(len(nus))

"""
Integration routines
"""
    
def integrate_multipop_nx(demo, Ttop, Tbot, rho, theta, y, ordered_pops, dt):
    # total integration time
    T = Tbot-Ttop
    # pops and models in this period, order from seeding from last epoch
    pops = ordered_pops
    npops = len(pops)
    models = [demo.node[pop]['model'] for pop in pops]
    if npops > 1:
        ms = get_migration_rates(demo,T,ordered_pops)
        M = migration_multipop(ms,npops)
    
    R = recombination_multipop(rho,npops)
    U = mutation_multipop(theta,npops)

    if models[0] == 'constant' and len(set(models)) == 1: # all constant sizes
        nus = [demo.node[pop]['nu'] for pop in pops]
        D = drift_multipop(nus,npops)
        
        if npops > 1:
            Ab = D+M+R+U
        else:
            Ab = D+R+U
        
        Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
        Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
        
        elapsed_T = Ttop
        # improve with t_elapsed, below checking if pop sizes changed
        while elapsed_T < Tbot:
            if elapsed_T + dt > Tbot:
                dt = Tbot-elapsed_T
                Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
                Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
            
            y = Ab2(Ab1.dot(y))
            elapsed_T += dt
    
    else:
        # nonconstant demography requires us to recompute the transition at each time step
        elapsed_T = Ttop
        
        while elapsed_T < Tbot:
            if elapsed_T + dt > Tbot:
                dt = Tbot-elapsed_T
            
            nus = get_pop_sizes(demo, elapsed_T+dt/2, ordered_pops)
            D = drift_multipop(nus,npops)
            
            if npops > 1:
                Ab = D+M+R+U
            else:
                Ab = D+R+U
            
            Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
            Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
            
            y = Ab2(Ab1.dot(y))
            elapsed_T += dt
    
    return y

#root_cache = {}
def root_equilibrium_nx(demo, rho, theta, dt):
    #try:
        #y0 = root_cache[(rho,theta)]
    #except KeyError:
    D = drift_multipop([1.],1)
    R = recombination_multipop(rho,1)
    U = mutation_multipop(theta,1)
    
    Ab = D+R+U
        
    y0 = np.array([0,0,1,1,1,1])
    Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
    Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
    for timesteps in range(int(20.0/dt)):
        y0 = Ab2(Ab1.dot(y0))
        
        #root_cache[(rho,theta)] = y0

    return y0, [get_root(demo)]

"""
Integration for multiple populations in style of moments
"""

def integrate_multipop(y, nu, T, num_pops=1, rho=0.0, theta=0.0008, dt=0.001, m=None):
    """
    Integration function for multipopulation statistics
    y: LDstats object with y.data, y.num_pops, y.order (=2 for multipop models)
    nu: 
    T: 
    rho: 
    theta: 
    dt: 
    m: migration matrix with [[0, m12, m13, ...],[m21, 0, m23, ...],...]
    
    Note that in the multipopulation basis, only the reversible mutation model is possible
    """
    moms = moment_names_multipop(num_pops)
    if len(moms)+1 != len(y):
        raise ValueError("num_pops must be set to correct number of populations")
    
    if num_pops > 1:
        if m== None:
            ms = num_pops*(num_pops-1)*[0]
        else:
            ms = []
            for ii in range(num_pops):
                for jj in range(ii+1,num_pops):
                    # note that in Matrices, we've reversed the meaning of m_ij (easier to fix here)
                    ms.append(m[jj][ii])
                    ms.append(m[ii][jj])
            M = migration_multipop(ms,num_pops)
    
    R = recombination_multipop(rho,num_pops)
    U = mutation_multipop(theta,num_pops)
    
    if callable(nu):
        nus = nu(0)
    else:
        nus = nu
    
    D = drift_multipop(nus,num_pops)
    
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
            if num_pops > 1:
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

def root_equilibrium(rho, theta, dt=0.01):
    D = drift_multipop([1.],1)
    R = recombination_multipop(rho,1)
    U = mutation_multipop(theta,1)
    
    Ab = D+R+U
        
    y0 = np.array([0,0,1,1,1,1])
    Ab1 = identity(Ab.shape[0], format='csc') + dt/2.*Ab
    Ab2 = factorized(identity(Ab.shape[0], format='csc') - dt/2.*Ab)
    for timesteps in range(int(20.0/dt)):
        y0 = Ab2(Ab1.dot(y0))
    return y0



    
