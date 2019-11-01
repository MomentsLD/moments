import networkx as nx
import numpy as np
import copy

from . import LDstats_mod
from . import Numerics

## check nx version (must be >= 2.1)
def check_nx_version():
    assert (float(nx.__version__) >= 2.0), "networkx must be version 2.0 or higher to use Demography"

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

nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs = ( 2.11, 
    0.377, 0.251, 0.111, 0.0904, 5.77, 0.224, 3.02, 0.0711, 3.80, 0.256, 0.125, 1.07 ) 
    
# Define a directed graph G
G = nx.DiGraph()

# The nodes are the populations (with attributes nu, T, etc), and directed edges are 
# population relationships (weights are 1 unless merger, in which case must sum to 1)
G.add_node('root', nu=1, T=0)
G.add_node('A', nu=nuA, T=TA)
G.add_node('B', nu=nuB, T=TB, m={'YRI': mAfB})
G.add_node('YRI', nu=nuA, T=TB+TF, m={'B': mAfB, 'CEU': mAfEu, 'CHB': mAfAs})
G.add_node('CEU', nu0=nuEu0, nuF=nuEuF,
                    T=TF, m={'YRI': mAfEu, 'CHB': mEuAs})
G.add_node('CHB', nu0=nuAs0, nuF=nuAsF,
                    T=TF, m={'YRI': mAfAs, 'CEU': mEuAs})

G.add_edges_from([ ('root', 'A'), 
                   ('A', 'YRI'), 
                   ('A', 'B'), 
                   ('B', 'CEU'), 
                   ('B', 'CHB') ])

### for mergers, we can use weighted directed edges, where the weights into any node sum to 1
### G.add_edge( (A, B), f)

For pulse migration events, we have pulse={('pop_to',x, f),('other_pop_to',x2,f2)}, 
    where x is the fraction along that node that pulse migration event occurs, 
    with fraction f replacement. So 0<x<1, e.g. if x=.2 and T=0.1, then the pulse 
    occurs at time 0.02 since the beginning of that branch.

We don't want every node to have to be split pre and post pulse events, which
    complicates situations with size changes. To specify exponential growth 
    models, we can pass either nu0 & nuF, nu0 & growth_rate, or nuF & 
    growth_rate, as any of these options uniquely define the growth function.
    This also removes relying the user to specify the lambda nu_func, which can
    be confusing, and sets us up for easier integration with msprime.
"""



def evolve(demo_graph, theta=0.001, rho=None, pop_ids=None):
    """
    demo_graph : a directed graph containing the topology and parameters of the demography
    theta : the scaled mutation rate 4*N*u
    rho : if None, we only compute heterozygosity statistics
          if scalar, we compute heterozygosity and LD statistics for that rho
          if list or array, we compute heterozygosity and LD statistics for each rho
    """
    check_nx_version()
    
    # get the features of the demography tree    
    root, parents, children, leaves = get_pcl(demo_graph)
    (present_pops, integration_times, nus, migration_matrices, frozen_pops, 
        selfing_rates, events) = get_event_times(demo_graph)
    
    # Initialize the demography at the root
    Y = LDstats_mod.LDstats(equilibrium(rho=rho, theta=theta), 
            num_pops=1, pop_ids=[present_pops[0][0]])
    if selfing_rates[0] is not None:
        Y.integrate([1.], 20, rho=rho, theta=theta, selfing=selfing_rates[0])
    
    # loop through epochs, integrating and applying demographic events 
    # such as splits and mergers
    for ii, (pops, T, nu, mig_mat, frozen, selfing) in enumerate(zip(
                    present_pops, integration_times, nus, 
                    migration_matrices, frozen_pops, selfing_rates)):
        # Instead of passing lambda functions, pass starting and ending sizes
        # across this integration interval. If every entry in nus is a single
        # value, just pass the list of nus. Otherwise, build the nu_func.
        nu_epoch = get_pop_size_function(nu)
        
        # Integrate for T
        Y.integrate(nu_epoch, T, rho=rho, theta=theta, 
                m=mig_mat, selfing=selfing, frozen=frozen)
        
        # Apply events that occur at the end of this epoch
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
                # we should have a catch to make sure marginalizations occur at 
                # end of events, since some events could include
                # those marginalized pops, and we want to keep them around
                for e in events[ii]:
                    if e[0] == 'split':
                        Y = dg_split(Y, e[1], e[2], e[3])
                    elif e[0] == 'merger':
                        Y = dg_merge(Y, e[1], e[2], e[3]) # pops_from, weights, pop_to
                    elif e[0] == 'pulse':
                        Y = dg_pulse(Y, e[1], e[2], e[3]) # pop_from, pop_to, f
                    elif e[0] == 'marginalize':
                        Y = Y.marginalize(Y.pop_ids.index(e[1])+1)
            # make sure correct order of pops for the next epoch
            Y = rearrange_pops(Y, present_pops[ii+1])
    
    # make sure populations are in the correct order as specified by pop_ids
    if pop_ids is not None:
        Y = rearrange_pops(Y, pop_ids)
    
    return Y 

def equilibrium(rho=None, theta=0.001):
    return Numerics.steady_state(theta=theta, rho=rho)

def get_pop_size_function(nus):
    """
    Every entry in nus is either a value (int/float) or list of length 2, as
        [nu0, growth_rate]
    growth_rate is found as r = np.log(nuF/nu0)/T, nu_func=nu0*exp(r*t)
    """
    nu = []
    any_fn = False
    for pop_nu in nus:
        if hasattr(pop_nu, '__len__'):
            any_fn = True
    
    if any_fn:
        for pop_nu in nus:
            if hasattr(pop_nu, '__len__'):
                any_fn = True
                nu0 = pop_nu[0]
                growth_rate = pop_nu[1]
                nu.append( lambda t, nu0=nu0, growth_rate=growth_rate: nu0 * np.exp(growth_rate*t) )
            else:
                nu.append( lambda t, pop_nu=pop_nu: pop_nu )
        return lambda t: [nu_func(t) for nu_func in nu]
    else:
        return nus

def dg_split(Y, parent, child1, child2):
    ids_from = Y.pop_ids
    Y = Y.split(ids_from.index(parent)+1)
    ids_to = ids_from + [child2]
    ids_to[ids_from.index(parent)] = child1
    Y.pop_ids = ids_to
    return Y

def dg_merge(Y, pops_to_merge, weights, pop_to):
    """
    Two populations (pops_to_merge = [popA, popB]) merge (with given weights)
    and form new population (pop_to).
    """
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

def dg_pulse(Y, pop_from, pop_to, pulse_weight):
    """
    A pulse migration event
    Different from merger, where the two parental populations are replaced by the 
    admixed population.
    Here, pulse events keep the current populations in place.
    """
    if pop_to in Y.pop_ids:
        ind_from = Y.pop_ids.index(pop_from)
        ind_to = Y.pop_ids.index(pop_to)
        Y = Y.pulse_migrate(ind_from+1, ind_to+1, pulse_weight)
    else:
        print("warning: pop_to in pulse_migrate isn't in present pops")
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
    if list(current_order) != list(pop_order):
        print("population ordering is fucked up")
        print(current_order)
        print(pop_order)
    return Y

def get_pcl(demo_graph):
    parents = {}
    for pop in demo_graph:
        if len(list(demo_graph.predecessors(pop))) == 0:
            root = pop
        else:
            # usually one parent, could be two in case of merger
            parents[pop] = list(demo_graph.predecessors(pop)) 
    
    # could have zero, one, or two children 
    # (if zero, node is a leaf and doesn't appear here)
    children = {} 
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
    try:
        selfing_rates = [[demo_graph.nodes[root]['selfing']]]
    except KeyError:
        selfing_rates = [None]
    events = []
    
    # tracks time left on each branch to integrate, of present_pops[-1]
    time_left = [0.] 
    
    # tracks time left on each branch to the next pulse event, if no pulse, set as None
    pulse_migration_events = get_pulse_events(demo_graph)
    
    advance = True
    while advance == True:
        # if no pop has any time left and all pops are leaves, end it
        if np.all(np.array(time_left) < tol) and np.all([p not in children for p in present_pops[-1]]):
            advance = False
        else:
            new_pops = [] # records populations present after any events that need to occur
            new_times = [] # records time left of these populations
            new_pulse_times = [] # records time to pulse events of these populations, 
            new_nus = []
            new_events = []
            
            # if any population is at an end, apply events
            # else update it's time left and nus
            for ii,pop_time_left in enumerate(time_left):
                this_pop = present_pops[-1][ii]
                if pop_time_left < tol:
                    if this_pop in children: # if it has children (1 or 2), split or carry over
                        # check if children already in new_pops, if so, it's a merger (with weights)
                        for child in children[this_pop]:
                            if child not in new_pops:
                                new_pops += children[this_pop]
                                new_times += [demo_graph.nodes[child]['T'] for child in children[this_pop]]
                                new_nus += [add_size_to_nus(demo_graph, child, demo_graph.nodes[child]['T']) for child in children[this_pop]]
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
                    
                    new_nus += [add_size_to_nus(demo_graph, this_pop, pop_time_left)]
            
            # for previous pops, check if any have a pulse occuring now
            # we'll update times directly in the pulse_migration_events dictionary
            for this_pop in present_pops[-1]:
                if this_pop in pulse_migration_events:
                    for pulse_event in pulse_migration_events[this_pop]:
                        if pulse_event[0] < 0: # this pulse already occurred
                            continue
                        elif pulse_event[0] < tol: # this pulse occurs now
                            new_events.append( ('pulse', this_pop, pulse_event[1], pulse_event[2]) )
            
            # for new pops, get the times to the next pulse (ones that are positive)
            # (we already set negative the times to pulse if they have occured)
            for this_pop in new_pops:
                if this_pop not in pulse_migration_events:
                    new_pulse_times.append(1e10)
                else:
                    temp_time = 1e10
                    for pulse_event in pulse_migration_events[this_pop]:
                        if pulse_event[0] > 0:
                            temp_time = min(temp_time, pulse_event[0])
                    new_pulse_times.append(temp_time)
            
            # set integration time of this epoch to next pulse or end of population
            time_left = new_times
            t_epoch = min(min(time_left), min(new_pulse_times))
            integration_times.append(t_epoch)
            
            # update times left to next events
            time_left = [pop_time_left - t_epoch for pop_time_left in time_left]
            pulse_migration_events = update_pulse_migration_events(pulse_migration_events, new_pops, t_epoch)
            
            present_pops.append(new_pops)
            
            nus.append(new_nus)
            
            # get the migration matrix for this epoch
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
            
            # get the list of frozen pops for this epoch
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
            
            # get selfing rates for this epoch
            selfing = []
            for pop in new_pops:
                if 'selfing' in demo_graph.nodes[pop]:
                    selfing.append(demo_graph.nodes[pop]['selfing'])
                else:
                    selfing.append(0)
            if set(selfing) == {0}:
                selfing_rates.append(None)
            else:
                selfing_rates.append(selfing)
            
            # rearrange new events so that marginalize happens last
            new_events_reordered = reorder_events(new_events)
            
            events.append(new_events_reordered)
    
    return present_pops, integration_times, nus, migration_matrices, frozen_pops, selfing_rates, events

def update_pulse_migration_events(pulse_migration_events, new_pops, t_epoch):
    for pop in new_pops:
        if pop in pulse_migration_events:
            for pulse_event in pulse_migration_events[pop]:
                pulse_event[0] -= t_epoch
    return pulse_migration_events

def get_pulse_events(demo_graph):
    # get all the pulse events for each branch, their times, pop_to, and weights 
    pulses = {}
    for pop_from in demo_graph.nodes:
        if 'pulse' in demo_graph.nodes[pop_from]:
            pulses[pop_from] = []
            for pulse_event in demo_graph.nodes[pop_from]['pulse']:
                pulse_time = pulse_event[1] * demo_graph.nodes[pop_from]['T']
                pop_to = pulse_event[0]
                weight = pulse_event[2]
                pulses[pop_from].append([pulse_time, pop_to, weight])
            # order them chronologically
            pulses[pop_from] = sorted(pulses[pop_from])
    return pulses

def reorder_events(new_events):
    """
    Place marginalize events at end of events
    """
    new_events_reordered = []
    for event in new_events:
        if event[0] != 'marginalize':
            new_events_reordered.append(event)
    for event in new_events:
        if event[0] == 'marginalize':
            new_events_reordered.append(event)
    return new_events_reordered
    
def add_size_to_nus(demo_graph, pop, time_left):
    """
    adds either nu, or [nu0, growth_rate], where nu0 is the size at the beginning of the epoch
    use time_left to set nu0 to the size at the beginning of the epoch
    """
    if 'nu' in demo_graph.nodes[pop]:
        return demo_graph.nodes[pop]['nu']
    else:
        tt = demo_graph.nodes[pop]['T'] - time_left
        if 'nu0' in demo_graph.nodes[pop] and 'nuF' in demo_graph.nodes[pop]:
            growth_rate = np.log(demo_graph.nodes[pop]['nuF']/demo_graph.nodes[pop]['nu0']) / demo_graph.nodes[pop]['T']
            nu0 = demo_graph.nodes[pop]['nu0'] * np.exp(growth_rate * tt)
            return [nu0, growth_rate]
        elif 'growth_rate' in demo_graph.nodes[pop] and 'nuF' in demo_graph.nodes[pop]:
            nu0_pop = demo_graph.nodes[pop]['nuF'] * np.exp(-demo_graph.nodes[pop]['growth_rate']*demo_graph.nodes[pop]['T'])
            nu0 = nu0_pop * np.exp(growth_rate * tt)
            return [nu0, demo_graph.nodes[pop]['growth_rate']]
        elif 'growth_rate' in demo_graph.nodes[pop] and 'nu0' in demo_graph.nodes[pop]:
            nu0 = demo_graph.nodes[pop]['nu0'] * np.exp(demo_graph.nodes[pop]['growth_rate'] * tt)
            return [nu0, demo_graph.nodes[pop]['growth_rate']]

def check_graph_validity(demo_graph):
    """
    Double check we've properly defined the demo_graph (that each node has all 
    required attributes, size functions have been properly defined, ...
    """
    pass
