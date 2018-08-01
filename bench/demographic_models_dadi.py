import numpy
import dadi


def model_YRI_CEU((nu1F, nu2B, nu2F, m, Tp, T), (n1,n2), pts):
    # Define the grid we'll use
    xx = yy = dadi.Numerics.default_grid(pts)

    # phi for the equilibrium ancestral population
    phi = dadi.PhiManip.phi_1D(xx)
    # Now do the population growth event.
    phi = dadi.Integration.one_pop(phi, xx, Tp, nu=nu1F)

    # The divergence
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    # We need to define a function to describe the non-constant population 2
    # size. lambda is a convenient way to do so.
    nu2_func = lambda t: nu2B*(nu2F/nu2B)**(t/T)
    phi = dadi.Integration.two_pops(phi, xx, T, nu1=nu1F, nu2=nu2_func, 
                                    m12=m, m21=m)
    # Finally, calculate the spectrum.
    sfs = dadi.Spectrum.from_phi(phi, (n1,n2), (xx,yy))
    return sfs

def model_YRI_CEU_extrap((nu1F, nu2B, nu2F, m, Tp, T), (n1,n2), (pts1, pts2, pts3)):
    """ Richardson extrapolation version of the model above"""
    model_extrap = dadi.Numerics.make_extrap_log_func(model_YRI_CEU)
    sfs = model_extrap((nu1F, nu2B, nu2F, m, Tp, T), (n1,n2), [pts1, pts2, pts3])
    return sfs


def model_ooa_3D((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu,
                 mAfAs, mEuAs, TAf, TB, TEuAs), (n1,n2,n3), pts):
    """A three-population model used to model out-of-Africa demography. swap describes the order in which we wish to output the Afr, Eur, Asian. (0,1,2) means [Afr, Eur, Asian], (1,2,0) means [Eur, As,Af] """

    xx = dadi.Numerics.default_grid(pts)
    #first step: a single population
    phi = dadi.PhiManip.phi_1D(xx)
    
    #integrate for time TAf (with constant population)
    phi = dadi.Integration.one_pop(phi, xx, TAf, nu=nuAf)
    #separate into two populations.
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    
    #integrate two populations (how are grid points chosen?)
    phi = dadi.Integration.two_pops(phi, xx, TB, nu1=nuAf, nu2=nuB,
                               m12=mAfB, m21=mAfB)
                               
    #split into three pops
    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx, phi)
    #define functions for population sizes
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    
    phi = dadi.Integration.three_pops(phi, xx, TEuAs, nu1=nuAf,
                                 nu2=nuEu_func, nu3=nuAs_func,
                                 m12=mAfEu, m13=mAfAs, m21=mAfEu, m23=mEuAs,
                                 m31=mAfAs, m32=mEuAs)

    sfs = dadi.Spectrum.from_phi(phi, (n1,n2,n3), (xx,xx,xx))
    
    return sfs

def model_ooa_3D_extrap((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu,
                        mAfAs, mEuAs, TAf, TB, TEuAs), (n1,n2,n3), (pts1, pts2, pts3)):
    """ Richardson extrapolation version of the model above"""
    model_extrap = dadi.Numerics.make_extrap_log_func(model_ooa_3D)
    sfs = model_extrap((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu,
                        mAfAs, mEuAs, TAf, TB, TEuAs), (n1,n2,n3), [pts1, pts2, pts3])
    return sfs
