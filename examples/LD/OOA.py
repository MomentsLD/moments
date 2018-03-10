import numpy as np
import moments.LD

# parameters for out of africa model
Taf = 0.265
Tb = 0.342 - 0.265
Tf = 0.405 - 0.342
nu_af = 1.98
nu_b = 0.255
nu_eu0 = 0.141
nu_as0 = 0.0752
nu_euf = 4.65
nu_asf = 6.22
m_af_b = 1.1
m_af_eu = 0.183
m_af_as = 0.57
m_eu_as = 0.227

p = (Taf,Tb,Tf,nu_af,nu_b,nu_eu0,nu_as0,nu_euf,nu_asf,m_af_b,m_af_eu,m_af_as,m_eu_as)

N0 = 7310
u = 2.36e-8
theta = 4*N0*u

def OutOfAfrica(params, rho=0.0, theta=0.0001):
    (Taf,Tb,Tf,nu_af,nu_b,nu_eu0,nu_as0,nu_euf,nu_asf,m_af_b,m_af_eu,m_af_as,m_eu_as) = params
    y = moments.LD.Numerics.root_equilibrium(rho,theta)
    y = moments.LD.LDstats(y)
    y.integrate([nu_af], Taf, rho=rho, theta=theta)
    y = y.split(1)
    y.integrate([nu_af,nu_b], Tb, rho=rho, theta=theta, m=[[0, m_af_b],[m_af_b, 0]])
    y = y.split(2)
    nu_func_eu = lambda t: nu_eu0 * (nu_euf/nu_eu0)**(t/Tf)
    nu_func_as = lambda t: nu_as0 * (nu_asf/nu_as0)**(t/Tf)
    nu_func = lambda t: [nu_af, nu_func_eu(t), nu_func_as(t)]
    y.integrate(nu_func, Tf, rho=rho, theta=theta, m=[[0, m_af_eu, m_af_as],[m_af_eu, 0, m_eu_as],[m_af_as, m_eu_as, 0]])
    return y


rho = 0.0

y = OutOfAfrica(p, rho=rho, theta=theta)

print y
print rho
print y[1]/np.sqrt(y[0]*y[3])
print y[2]/np.sqrt(y[0]*y[5])
print y[4]/np.sqrt(y[3]*y[5])

####

import networkx as nx

def YRI_CEU_nx(params, rho=0, theta=1e-4, dt=0.001):
    Taf,Tf,nu_af,nu_eu0,nu_euf,m = params
    demo = nx.DiGraph()
    demo.add_nodes_from(['Af','Eu','Af-Eu'])
    demo.add_edges_from([('Af-Eu','Af'),('Af-Eu','Eu')])
    
    tree_times = {'Af-Eu' : Taf, 'F' : Taf+Tf} 
    
    # add time attributes to each node
    demo = moments.LD.Numerics.add_times(demo, tree_times)

    # add model types along each branch. right now simple
    demo.node['Af-Eu']['model'] = 'constant'
    demo.node['Af-Eu']['nu'] = nu_af
    demo.node['Af']['model'] = 'constant'
    demo.node['Af']['nu'] = nu_af
    demo.node['Eu']['model'] = 'exponential'
    demo.node['Eu']['nu_top'] = nu_eu0
    demo.node['Eu']['nu_bot'] = nu_euf

    # add migration rates
    for pop in moments.LD.Numerics.get_interiors(demo)+moments.LD.Numerics.get_leaves(demo):
        demo.node[pop]['mig'] = {}

    demo.node['Af']['mig']['Eu'] = m
    demo.node['Eu']['mig']['Af'] = m
    
    y, pops = moments.LD.Numerics.demography(demo, rho=rho, theta=theta, dt=dt)
    return y

def YRI_CEU(params, rho=0, theta=1e-4, dt=0.001):
    Taf,Tf,nu_af,nu_eu0,nu_euf,m = params
    y = moments.LD.Numerics.root_equilibrium(rho,theta)
    y = moments.LD.LDstats(y)
    y.integrate([nu_af], Taf, rho=rho, theta=theta, dt=dt)
    y = y.split(1)
    nu_func_eu = lambda t: nu_eu0 * (nu_euf/nu_eu0)**(t/Tf)
    nu_func = lambda t: [nu_af, nu_func_eu(t)]
    y.integrate(nu_func, Tf, rho=rho, theta=theta, m=[[0, m],[m, 0]], dt=dt)
    return y


params = (0.2, 0.5, 2.0, 0.1, 4.0, 1.0)

y1 = YRI_CEU_nx(params, rho=0.0, theta=0.000576, dt=0.001)
y2 = YRI_CEU(params, rho=0.0, theta=0.000576, dt=0.001)



#### simpler model

import networkx as nx

def YRI_CEU_const_nx(params, rho=0, theta=1e-4, dt=0.001):
    Taf, Tf, nu_af, nu_eu, m = params
    demo = nx.DiGraph()
    demo.add_nodes_from(['Af','Eu','Af-Eu'])
    demo.add_edges_from([('Af-Eu','Af'),('Af-Eu','Eu')])
    
    tree_times = {'Af-Eu' : Taf, 'F' : Taf+Tf} 
    
    # add time attributes to each node
    demo = moments.LD.Numerics.add_times(demo, tree_times)

    # add model types along each branch. right now simple
    demo.node['Af-Eu']['model'] = 'constant'
    demo.node['Af-Eu']['nu'] = nu_af
    demo.node['Af']['model'] = 'constant'
    demo.node['Af']['nu'] = nu_af
    demo.node['Eu']['model'] = 'constant'
    demo.node['Eu']['nu'] = nu_eu

    # add migration rates
    for pop in moments.LD.Numerics.get_interiors(demo)+moments.LD.Numerics.get_leaves(demo):
        demo.node[pop]['mig'] = {}

    demo.node['Af']['mig']['Eu'] = m
    demo.node['Eu']['mig']['Af'] = m
    
    y, pops = moments.LD.Numerics.demography(demo, rho=rho, theta=theta, dt=dt)
    return y

def YRI_CEU_const(params, rho=0, theta=1e-4, dt=0.001):
    Taf, Tf, nu_af, nu_eu, m = params
    y = moments.LD.Numerics.root_equilibrium(rho,theta)
    y = moments.LD.LDstats(y)
    y.integrate([nu_af], Taf, rho=rho, theta=theta, dt=dt)
    y = y.split(1)
    y.integrate([nu_af,nu_eu], Tf, rho=rho, theta=theta, m=[[0, m],[m, 0]], dt=dt)
    return y


params = (0.2, 0.5, 1.0, 1.0, 0.0)

y1 = YRI_CEU_const_nx(params, rho=0.0, theta=0.000576, dt=0.001)
y2 = YRI_CEU_const(params, rho=0.0, theta=0.000576, dt=0.001)
