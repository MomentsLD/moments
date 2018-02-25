import numpy as np
import cPickle as pickle
import networkx as nx
import moments.LD

# We'll test it out with a three population demography 
# and later will automate the computation
#       |
#       | root '1-2-3'
#      / \ 
#     /   \ internal branch '2-3'
#    /    /\
#   /    /  \
#   1    2  3

# interior populations must start with 'i', and times much match
# so if an interior pop is named 'doggy', its split time is named 'T_doggy'


### simple two population split + migration, constant size demography
def split_mig_2pop(params, rho=0, theta=1e-4, dt=0.01):
    T,nu1,nu2,m12,m21 = params
    demo = nx.DiGraph()
    demo.add_nodes_from(['1','2','1-2'])
    demo.add_edges_from([('1-2','1'),('1-2','2')])
    # tree_times needs all the split times and the simulation end time
    tree_times = {'1-2' : 0.0,  # '1-2' splits into 1 and 2 at time T = 0
                  'F' : T}  # simulation ends at T = 0.5
    # fill in event times
    demo = moments.LD.Numerics.add_times(demo,tree_times)
    # fill in population models and sizes
    demo.node['1-2']['model'] = 'constant'
    demo.node['1-2']['nu'] = 1.0
    demo.node['1']['model'] = 'constant'
    demo.node['1']['nu'] = nu1
    demo.node['2']['model'] = 'constant'
    demo.node['2']['nu'] = nu2
    # add migration rates
    for pop in moments.LD.Numerics.get_interiors(demo)+moments.LD.Numerics.get_leaves(demo):
        demo.node[pop]['mig'] = {}
    demo.node['1']['mig']['2'] = m12
    demo.node['2']['mig']['1'] = m21
    y, pops = moments.LD.Numerics.demography(demo, rho=rho, theta=theta, dt=dt)
    return y, pops

# model params
T = 0.5
nu1 = 1.0
nu2 = 1.0
m12 = m21 = 0.0

edges = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
mids = (edges[:-1]+edges[1:])/2

N = 1e4
u = 1.44e-8

(n1,n2) = (10,20)

theta = 4 * N * u
p = (T,nu1,nu2,m12,m21) 

stats_edges = []
stats_mids = []

for rho in edges:
    y, pops = split_mig_2pop(p,rho=rho,theta=theta,dt=0.001)
    stats_edges.append(moments.LD.Corrections.corrected_multipop(2, (n1,n2), y))

for rho in mids:
    y, pops = split_mig_2pop(p,rho=rho,theta=theta,dt=0.001)
    stats_mids.append(moments.LD.Corrections.corrected_multipop(2, (n1,n2), y))

stats = []
for ii in range(len(mids)):
    stats.append(1./6*(stats_edges[ii]+stats_edges[ii+1]+4*stats_mids[ii]))


stats_1D_ns1 = []
stats_1D_ns2 = []

stats_edges_ns1 = []
stats_mids_ns1 = []
stats_edges_ns2 = []
stats_mids_ns2 = []

for rho in edges:
    y = moments.LD.Demographics.two_epoch(2,(nu1,T),rho=rho,theta=theta,corrected=True,ns=n1)
    stats_edges_ns1.append(y[:-1])
    y = moments.LD.Demographics.two_epoch(2,(nu1,T),rho=rho,theta=theta,corrected=True,ns=n2)
    stats_edges_ns2.append(y[:-1])

for rho in mids:
    y = moments.LD.Demographics.two_epoch(2,(nu1,T),rho=rho,theta=theta,corrected=True,ns=n1)
    stats_mids_ns1.append(y[:-1])
    y = moments.LD.Demographics.two_epoch(2,(nu1,T),rho=rho,theta=theta,corrected=True,ns=n2)
    stats_mids_ns2.append(y[:-1])

for ii in range(len(mids)):
    stats_1D_ns1.append(1./6*(stats_edges_ns1[ii]+stats_edges_ns1[ii+1]+4*stats_mids_ns1[ii]))
    stats_1D_ns2.append(1./6*(stats_edges_ns2[ii]+stats_edges_ns2[ii+1]+4*stats_mids_ns2[ii]))


#
## do these match when migration is set to zero? yes
#
### implementation of Gravel Out of Africa model
#import numpy as np
#import pickle
#import networkx as nx
#import numerics_multipop
#reload(numerics_multipop)
#
#def OutOfAfrica(params, rho=0, theta=1e-4, dt=0.01):
#    Taf,Tb,Tf,nu_af,nu_b,nu_eu0,nu_as0,nu_euf,nu_asf,m_af_b,m_af_eu,m_af_as,m_eu_as = params
#    demo = nx.DiGraph()
#    demo.add_nodes_from(['Af','Eu','As','Eu-As','Af-Eu-As'])
#    demo.add_edges_from([('Af-Eu-As','Af'),('Af-Eu-As','Eu-As'),('Eu-As','Eu'),('Eu-As','As')])
#    
#    tree_times = {'Af-Eu-As' : Taf, 'Eu-As' : Tb, 'F' : Tf} 
#    
#    # add time attributes to each node
#    demo = numerics_multipop.add_times(demo, tree_times)
#
#    # add model types along each branch. right now simple
#    demo.node['Af-Eu-As']['model'] = 'constant'
#    demo.node['Af-Eu-As']['nu'] = nu_af
#    demo.node['Eu-As']['model'] = 'constant'
#    demo.node['Eu-As']['nu'] = nu_b
#    demo.node['Af']['model'] = 'constant'
#    demo.node['Af']['nu'] = nu_af
#    demo.node['Eu']['model'] = 'exponential'
#    demo.node['Eu']['nu_top'] = nu_eu0
#    demo.node['Eu']['nu_bot'] = nu_euf
#    demo.node['As']['model'] = 'exponential'
#    demo.node['As']['nu_top'] = nu_as0
#    demo.node['As']['nu_bot'] = nu_asf
#
#    # add migration rates
#    for pop in numerics_multipop.get_interiors(demo)+numerics_multipop.get_leaves(demo):
#        demo.node[pop]['mig'] = {}
#
#    demo.node['Af']['mig']['Eu-As'] = m_af_b
#    demo.node['Af']['mig']['Eu'] = m_af_eu
#    demo.node['Af']['mig']['As'] = m_af_as
#    demo.node['Eu-As']['mig']['Af'] = m_af_b
#    demo.node['Eu']['mig']['Af'] = m_af_eu
#    demo.node['Eu']['mig']['As'] = m_eu_as
#    demo.node['As']['mig']['Af'] = m_af_as
#    demo.node['As']['mig']['Eu'] = m_eu_as
#    
#    y, pops = numerics_multipop.demography(demo, rho=rho, theta=theta, dt=dt)
#    return y, pops
#
#Taf = 0.265
#Tb = 0.342
#Tf = 0.405
#nu_af = 1.98
#nu_b = 0.255
#nu_eu0 = 0.141
#nu_as0 = 0.0752
#nu_euf = 4.65
#nu_asf = 6.22
#m_af_b = 1.1
#m_af_eu = 0.183
#m_af_as = 0.57
#m_eu_as = 0.227
#
#p = (Taf,Tb,Tf,nu_af,nu_b,nu_eu0,nu_as0,nu_euf,nu_asf,m_af_b,m_af_eu,m_af_as,m_eu_as)
#
#rho = 0.01
#
#N0 = 7310
#u = 2.36e-8
#theta = 4*N0*u
#
#y, pops = OutOfAfrica(p, rho=rho, theta=theta, dt=0.001)
#
##print pops
##print y
#print rho
#print y[1]/np.sqrt(y[0]*y[3])
#print y[2]/np.sqrt(y[0]*y[5])
#print y[4]/np.sqrt(y[3]*y[5])
#
