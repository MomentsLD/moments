import numpy
from dadi import Numerics, PhiManip, Integration, Spectrum

def OutOfAfrica((nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, 
                 mAfB, mAfEu, mAfAs, mEuAs, TAf, TB, TEuAs), (n1,n2,n3), pts):
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(phi, xx, TB, nu1=nuAf, nu2=nuB, 
                               m12=mAfB, m21=mAfB)

    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)

    nuEu_func = lambda t: nuEu0*(nuEu/nuEu0)**(t/TEuAs)
    nuAs_func = lambda t: nuAs0*(nuAs/nuAs0)**(t/TEuAs)
    phi = Integration.three_pops(phi, xx, TEuAs, nu1=nuAf, 
                                 nu2=nuEu_func, nu3=nuAs_func, 
                                 m12=mAfEu, m13=mAfAs, m21=mAfEu, m23=mEuAs,
                                 m31=mAfAs, m32=mEuAs)

    fs = Spectrum.from_phi(phi, (n1,n2,n3), (xx,xx,xx))
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
