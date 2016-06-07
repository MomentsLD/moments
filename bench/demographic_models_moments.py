import numpy
import moments

def model_ooa_3D((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, mAfB, mAfEu,
                 mAfAs, mEuAs, TAf, TB, TEuAs), (n1,n2,n3)):
    """A three-population model used to model out-of-Africa demography.
    """
    #first step: a single population
    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3)
    fs = moments.Spectrum(sts)
    #integrate for time TAf (with constant population)
    fs.integrate([nuAf], [n1+n2+n3], TAf, 0.05)
    
    #separate into two populations.
    fs = moments.Manips.split_1D_to_2D(fs, n1, n2+n3)
    
    #integrate two populations
    # migration rates matrix
    mig1=numpy.array([[0, mAfB],[mAfB, 0]])
    fs.integrate([nuAf, nuB], [n1, n2+n3], TB, 0.05, m=mig1)
    
    #split into three pops
    fs = moments.Manips.split_2D_to_3D_2(fs, n2, n3)

    #define functions for population sizes
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    nu2 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t)]
    # migration rates matrix
    mig2=numpy.array([[0, mAfEu, mAfAs],[mAfEu, 0, mEuAs],[mAfAs, mEuAs, 0]])
    fs.integrate(nu2, [n1, n2, n3], TEuAs, 0.05, m=mig2)
                                
    return fs