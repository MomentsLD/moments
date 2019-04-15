import networkx as nx
import numpy as np
import copy

from . import LDstats_mod
from . import Numerics

tol = 1e-12
"""
We define a demography on an acyclic directed graph, where each node is a population
and each edge denotes parental relationship between populations. This can include 
splits, mergers, migrations, and size changes.
Each node (population) has a set of attributes. These are a population size or size 
function, migration rates to and from (not necessarily equal) other populations, and 
other possible attributes (e.g. frozen, for non-contemporary samples).

It helps to draw the treep topology with pen(cil) and paper, if there are many populations.

The evolve function takes in a demography graph object, and you specify the mutation
and recombination rates (theta and rho, resp). You can also specify if working with 
genotype data and whether to correct for sampling bias for a given sample size (ns).

Other functions extract the timing of events from the demo_graph, ...



# Example: Gutenkunst demographic model
   \\ root
    \\_
    / / A
   / /___
  / /--- \ B
 / /   //\\
/ /   //  \\
YRI  CEU  CHB

nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs = ( 2.11, 0.377, 0.251, 0.111, 0.0904, 5.77, 0.224, 3.02, 0.0711, 3.80, 0.256, 0.125, 1.07 ) 
    
# Define a directed graph G
G = nx.DiGraph()

# The nodes are the populations (with attributes nu, T, etc), and directed edges are population relationships (weights are 1 unless merger, in which case must sum to 1)
G.add_node('root', nu=1, T=0)
G.add_node('A', nu=nuA, T=TA)
G.add_node('B', nu=nuB, T=TB, m={'YRI': mAfB})
G.add_node('YRI', nu=nuA, T=TB+TF, m={'B': mAfB, 'CEU': mAfEu, 'CHB': mAfAs})
G.add_node('CEU', nu=lambda t: nuEu0 * np.exp(np.log(nuEuF/nuEu0) * t/TF ), 
                    T=TF, m={'YRI': mAfEu, 'CHB': mEuAs})
G.add_node('CHB', nu=lambda t: nuAs0 * np.exp(np.log(nuAsF/nuAs0) * t/TF ), 
                    T=TF, m={'YRI': mAfAs, 'CEU': mEuAs})

G.add_edges_from([ ('root', 'A'), 
                   ('A', 'YRI'), 
                   ('A', 'B'), 
                   ('B', 'CEU'), 
                   ('B', 'CHB') ])

### for mergers, we can use weighted directed edges, where the weights into any node sum to 1
### G.add_edge( (A, B), f)


"""



def evolve(demo_graph, theta=0.001, rho=None, pop_ids=None):
    """
    demo_graph : a directed graph containing the topology and parameters of the demography
    theta : the scaled mutation rate 4*N*u
    rho : if None, we only compute heterozygosity statistics
          if scalar, we compute heterozygosity and LD statistics for that rho
          if list or array, we compute heterozygosity and LD statistics for each rho
    """
    
#    # list of rhos, if we compute LD statistics
#    if rho == None:
#        continue
#    elif np.isscalar(rho):
#        rho = [rho]
    
    # set y (two locus stats) and h (heterozygosity stats) for ancestral pop
    
    root, parents, children, leaves = get_pcl(demo_graph)
    present_pops, integration_times, nus, migration_matrices, frozen_pops, events = get_event_times(demo_graph)
    
    Y = LDstats_mod.LDstats(equilibrium(rho=rho, theta=theta), num_pops=1, pop_ids=[present_pops[0][0]])
    
    for ii, (pops, T, nu, mig_mat, frozen) in enumerate(zip(present_pops, integration_times, nus, migration_matrices, frozen_pops)):
        if np.any([callable(nus[ii][i]) for i in range(len(nus[ii]))]):
            callable_nu = []
            for pop_nu in nu:
                if callable(pop_nu):
                    callable_nu.append( pop_nu )
                else:
                    callable_nu.append( lambda t, pop_nu=pop_nu: pop_nu)
            pass_nu = lambda t: [nu_func(t) for nu_func in callable_nu]
            Y.integrate(pass_nu, T, rho=rho, theta=theta, m=mig_mat, frozen=frozen)
        else:
            Y.integrate(nu, T, rho=rho, theta=theta, m=mig_mat, frozen=frozen)
        
        if ii < len(events):
            if len(events[ii]) ==  0:
                # just pass on populations, make sure keeping correct order of pop_ids
                new_ids = []
                for pid in pops:
                    if pid in present_pops[ii+1]:
                        new_ids.append(pid)
                    else:
                        new_ids.append(children[pid][0])
                Y.pop_ids = new_ids
            else:
                # apply events in between epochs
                for e in events[ii]:
                    if e[0] == 'split':
                        Y = dg_split(Y, e[1], e[2], e[3])
                    elif e[0] == 'merger':
                        Y = dg_merge(Y, e[1], e[2], e[3]) # pops_from, weights, pop_to
                    elif e[0] == 'marginalize':
                        Y = Y.marginalize(Y.pop_ids.index(e[1])+1)
            # make sure correct order of pops for the next epoch
            Y = rearrange_pops(Y, present_pops[ii+1])
    
    if pop_ids is not None:
        Y = rearrange_pops(Y, pop_ids)
    
    return Y 

def equilibrium(rho=None, theta=0.001):
    return Numerics.steady_state(theta=theta, rho=rho)

def dg_split(Y, parent, child1, child2):
    ids_from = Y.pop_ids
    Y = Y.split(ids_from.index(parent)+1)
    ids_to = ids_from + [child2]
    ids_to[ids_from.index(parent)] = child1
    Y.pop_ids = ids_to
    return Y

def dg_merge(Y, pops_to_merge, weights, pop_to):
    pop1,pop2 = pops_to_merge
    ids_from = Y.pop_ids
    ids_to = copy.copy(ids_from)
    ids_to.pop(ids_to.index(pop1))
    ids_to.pop(ids_to.index(pop2))
    ids_to.append(pop_to)
    pop1_ind = ids_from.index(pop1)
    pop2_ind = ids_from.index(pop2)
    if pop1_ind < pop2_ind:
        Y = Y.merge(pop1_ind+1, pop2_ind+1, weights[0])
    else:
        Y = Y.merge(pop2_ind+1, pop1_ind+1, weights[1])
    Y.pop_ids = ids_to
    return Y
    
def rearrange_pops(Y, pop_order):
    current_order = Y.pop_ids
    for i in range(len(pop_order)):
        if current_order[i] != pop_order[i]:
            j = current_order.index(pop_order[i])
            while j > i:
                Y = Y.swap_pops(j,j+1)
                current_order = Y.pop_ids
                j -= 1
    if current_order != pop_order:
        print("fucked up")
        print(current_order)
        print(pop_order)
    return Y

def get_pcl(demo_graph):
    parents = {}
    for pop in demo_graph:
        if len(list(demo_graph.predecessors(pop))) == 0:
            root = pop
        else:
            parents[pop] = list(demo_graph.predecessors(pop)) # usually one parent, could be two in case of merger
    
    children = {} # could have zero, one, or two children (if zero, node is a leaf and doesn't appear here)
    for pop in parents:
        for parent in parents[pop]: 
            children.setdefault(parent,[]) 
            children[parent].append(pop) 
    
    leaves = []
    for pop in demo_graph: 
        if pop not in children: 
            leaves.append(pop)
    
    return root, parents, children, leaves

def get_event_times(demo_graph):
    """
    For purposes of integration, we set time=0 to be the pre-event time in the ancestral
    population. Then every event (split, param change, merger) is a time since this 
    reference time. The last time in the returned list is "present" or the stopping 
    time for integration.
    """
    root, parents, children, leaves = get_pcl(demo_graph)
    
    present_pops = [[root]]
    integration_times = [demo_graph.nodes[root]['T']]
    nus = [[demo_graph.nodes[root]['nu']]]
    migration_matrices = [[0]]
    frozen_pops = [[False]]
    events = []
    
    time_left = [0.] # tracks time left on each branch to integrate, of present_pops[-1]
    advance = True
    while advance == True:
        # if no pop has any time left and all pops are leaves, end it
        if np.all(np.array(time_left) < tol) and np.all([p not in children for p in present_pops[-1]]):
            advance = False
        else:
            new_pops = []
            new_times = []
            new_nus = []
            new_events = []
            for ii,pop_time_left in enumerate(time_left):
                this_pop = present_pops[-1][ii]
                if pop_time_left < tol:
                    if this_pop in children: # if it has children (1 or 2), split or carry over
                        # check if children already in new_pops, if so, it's a merger (with weights)
                        for child in children[this_pop]:
                            if child not in new_pops:
                                new_pops += children[this_pop]
                                new_times += [demo_graph.nodes[child]['T'] for child in children[this_pop]]
                                new_nus += [demo_graph.nodes[child]['nu'] for child in children[this_pop]]
                        if len(children[this_pop]) == 2:
                            child1, child2 = children[this_pop]
                            new_events.append( ('split', this_pop, child1, child2) )
                        else:
                            child = children[this_pop][0]
                            # if the one child is a merger, need to specify, otherwise, nothing needed
                            if parents[child] == [this_pop]:
                                continue
                            else:
                                parent1, parent2 = parents[child]
                                # check if we've already recorded this event from the other parent
                                event_recorded = 0
                                for event in new_events:
                                    if event[0] == 'merger' and event[1] == (parent1, parent2):
                                        event_recorded = 1
                                if event_recorded == 1:
                                    continue
                                else:
                                    weights = (demo_graph.get_edge_data(parent1,child)['weight'], demo_graph.get_edge_data(parent2,child)['weight'])
                                    new_events.append( ('merger', (parent1, parent2), weights, child) )
                    else: # else no children and we eliminate it
                        new_events.append( ('marginalize', this_pop) )
                        continue
                else:
                    new_pops += [this_pop]
                    new_times += [pop_time_left]
                    
                    # if population is still here and if nu is callable (e.g. growth), need to reset time so that the function continues smoothly instead of resetting
                    if callable(demo_graph.nodes[this_pop]['nu']) == False:
                        new_nus += [demo_graph.nodes[this_pop]['nu']]
                    else:
                        new_nus.append( lambda t, this_pop=this_pop, pop_time_left=pop_time_left: demo_graph.nodes[this_pop]['nu']( demo_graph.nodes[this_pop]['T'] - pop_time_left + t ) )
                    
            time_left = new_times
            t_epoch = min(time_left)
            integration_times.append(t_epoch)
            time_left = [pop_time_left - t_epoch for pop_time_left in time_left]
            present_pops.append(new_pops)
            
            # check if any new_nus are callable, and if so make new_nus callable
#            nus_callable = 0
#            for nu in new_nus:
#                if callable(nu):
#                    nus_callable = 1
#            if nus_callable == 0:
#                nus.append(new_nus)
#            else:
#                for ii,nu in enumerate(new_nus):
#                    if callable( nu ):
#                        new_nus[ii] = nu 
#                    else:
#                        new_nus[ii] = lambda t: nu
#                nus.append( lambda t: [nu_func(t) for nu_func in new_nus] )
            nus.append(new_nus)
            
            new_m = np.zeros((len(new_pops), len(new_pops)))
            for ii, pop_from in enumerate(new_pops):
                if 'm' in demo_graph.nodes[pop_from]:
                    for jj, pop_to in enumerate(new_pops):
                        if pop_from == pop_to:
                            continue
                        else:
                            if pop_to in demo_graph.nodes[pop_from]['m']:
                                new_m[ii][jj] = demo_graph.nodes[pop_from]['m'][pop_to]
            
            migration_matrices.append(new_m)
            
            frozen = []
            for pop in new_pops:
                if 'frozen' in demo_graph.nodes[pop]:
                    if demo_graph.nodes[pop]['frozen'] == True:
                        frozen.append(True)
                    else:
                        frozen.append(False)
                else:
                    frozen.append(False)
            frozen_pops.append(frozen)
            
            events.append(new_events)
    
    return present_pops, integration_times, nus, migration_matrices, frozen_pops, events
    


