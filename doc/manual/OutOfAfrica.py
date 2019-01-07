import numpy
import moments

def OutOfAfrica((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, 
                 mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs), (n1,n2,n3)):
    """
    A three-population model used to model out-of-Africa demography.
    """
    #first step: a single population
    sts = moments.LinearSystem_1D.steady_state_1D(n1+n2+n3)
    fs = moments.Spectrum(sts)
    #integrate for time TAf (with constant population)
    fs.integrate([nuAf], TAf, 0.05)
    
    #separate into two populations.
    fs = moments.Manips.split_1D_to_2D(fs, n1, n2+n3)
    
    #integrate two populations
    # migration rates matrix
    mig1=numpy.array([[0, mAfB],[mAfB, 0]])
    fs.integrate([nuAf, nuB], TB, 0.05, m=mig1)
    
    #split into three pops
    fs = moments.Manips.split_2D_to_3D_2(fs, n2, n3)

    #define functions for population sizes
    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    nu2 = lambda t: [nuAf, nuEu_func(t), nuAs_func(t)]

    # migration rates matrix
    mig2=numpy.array([[0, mAfEu, mAfAs],[mAfEu, 0, mEuAs],[mAfAs, mEuAs, 0]])
    fs.integrate(nu2, TEuAs, 0.05, m=mig2)
                                
    return fs

def OutOfAfrica_mscore((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs,
                        mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs)):

    alphaEu = numpy.log(nuEu/nuEu0)/TEuAs
    alphaAs = numpy.log(nuAs/nuAs0)/TEuAs

    command = "-n 1 %(nuAf)f -n 2 %(nuEu)f -n 3 %(nuAs)f "\
            "-eg 0 2 %(alphaEu)f -eg 0 3 %(alphaAs)f "\
            "-ma x %(mAfEu)f %(mAfAs)f %(mAfEu)f x %(mEuAs)f %(mAfAs)f %(mEuAs)f x "\
            "-ej %(TEuAs)f 3 2 -en %(TEuAs)f 2 %(nuB)f "\
            "-ema %(TEuAs)f 3 x %(mAfB)f x %(mAfB)f x x x x x "\
            "-ej %(TB)f 2 1 "\
            "-en %(TAf)f 1 1"

    sub_dict = {'nuAf':nuAf, 'nuEu':nuEu, 'nuAs':nuAs,
                'alphaEu':2*alphaEu, 'alphaAs':2*alphaAs,
                'mAfEu': 2*mAfEu, 'mAfAs':2*mAfAs, 'mEuAs':2*mEuAs,
                'TEuAs': TEuAs/2, 'nuB': nuB, 'mAfB': 2*mAfB,
                'TB': (TEuAs+TB)/2., 'TAf': (TEuAs+TB+TAf)/2.}

    return command % sub_dict
