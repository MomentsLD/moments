import networkx as nx
import numpy as np

import moments.LD


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




"""

def get_event_times(demo_graph):
    """
    For purposes of integration, we set time=0 to be the pre-event time in the ancestral
    population. Then every event (split, param change, merger) is a time since this 
    reference time. The last time in the returned list is "present" or the stopping 
    time for integration.
    """
    pass

def get_pops_at_time(demo_graph, t):
    pass

def get_pop_sizes(demo_graph, event_times, t):
    if t in event_times:
        # we're at a t0
        pass
    else:
        pass

def get_migration_rates():
    pass

def get_epochs(demo_graph):
    event_times = get_event_times(demo_graph)
    pops = [get_pops_at_time(demo_graph, t) for t in ]
    t0s = event_times[:-1]
    Ts = event_times[1:] - event_times[:-1]
    nus = [ for epoch in ]
    ms = [ for epoch in ]
    return pops, Ts, nus, ms

def get_actions():
    # maybe we want get epochs and get actions together. and just step forward through the demography

def evolve(demo_graph, theta=0.001, rho=0.0, genotypes=False, ns=None):
    """
    
    """
    
    
    
    # set y (two locus stats) and h (heterozygosity stats) for ancestral pop
    y,h = 

    # two lists, one of actions (split, admix, param_change/none) at each epoch bdry, 
    # the other of
    # the "epochs" which store lists of integration times, pop sizes, ms, needed
    # to run each integration in turn, with the actions interspersed. 
    # if there are n epochs, then the will be n+1 actions (since we can have an
    # action at both beginning and end)
    
    












    pass


