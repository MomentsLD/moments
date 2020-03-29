import os, unittest

import numpy
import moments, moments.TwoLocus
import time


class TwoLocusTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_to_from_file(self):
        # make some data
        comments = ['comment 1', 'comment 2']
        filename = 'test.fs'
        data = numpy.random.rand(11,11,11)
        fs = moments.TwoLocus.TLSpectrum(data)
        # test saving and reloading data fs
        fs.to_file(filename, comment_lines=comments)
        fs_from = moments.TwoLocus.TLSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs, fs_from))
        
        # test saving with folding attribute
        fs_folded = fs.fold()
        fs_folded.to_file(filename)
        fs_from = moments.TwoLocus.TLSpectrum.from_file(filename)
        os.remove(filename)
        self.assertTrue(numpy.allclose(fs_folded, fs_from))
            
#    def test_projection(self):
#        fs40 = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns40_rho1.fs')
#        fs30 = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns30_rho1.fs')
#        fs_proj = fs40.project(30)
#        self.assertTrue(numpy.allclose(fs_proj, fs30, atol=1e-3))

## the jackknife test is commented because it takes a while for larger sample sizes (which is
## why I cache jk operators), and I haven't yet cythonized the Jackknife.py methods    
#    def test_twolocus_jackknife(self):
#        fs = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns40_rho1.fs')
#        fs1 = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns41_rho1.fs')
#        fs2 = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns42_rho1.fs')
#        ns = 40
#        J1 = moments.TwoLocus.Jackknife.calc_jk(ns,1)
#        J2 = moments.TwoLocus.Jackknife.calc_jk(ns,2)
#        jk1 = moments.TwoLocus.TLSpectrum( moments.TwoLocus.Numerics.Phi_to_array(
#                    J1.dot(moments.TwoLocus.Numerics.array_to_Phi(fs.data)), ns+1) )
#        jk2 = moments.TwoLocus.TLSpectrum( moments.TwoLocus.Numerics.Phi_to_array(
#                    J2.dot(moments.TwoLocus.Numerics.array_to_Phi(fs.data)), ns+2) )
#        
#        self.assertTrue(numpy.allclose(fs1, jk1, atol=2e-3))
#        self.assertTrue(numpy.allclose(fs2, jk2, atol=5e-3))

    def test_neutral_slow(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns{0}_rho0.fs'.format(ns))
        fs = moments.TwoLocus.TLSpectrum(numpy.zeros((ns+1,ns+1,ns+1)))
        fs.integrate(1, 20, rho=0.0)
        self.assertTrue(numpy.allclose(fs.data, cached.data))

    def test_recombination_slow(self):
        ns = 30
        cached = moments.TwoLocus.TLSpectrum.from_file('test_files/two_locus_ns{0}_rho1.fs'.format(ns))
        fs = moments.TwoLocus.TLSpectrum(numpy.zeros((ns+1,ns+1,ns+1)))
        fs.integrate(1, 20, rho=1.0)
        self.assertTrue(numpy.allclose(fs, cached))
    
#    def test_selection_slow(self):
#        # test if sel_params and gamma give same answer
#        ns = 30
#        gamma = -1.
#        sel_params = (-1.,-1.,0)
#        fs_gamma = moments.TwoLocus.TLSpectrum(numpy.zeros((ns+1,ns+1,ns+1)))
#        fs_sel_params = moments.TwoLocus.TLSpectrum(numpy.zeros((ns+1,ns+1,ns+1)))
#        fs_gamma.integrate(1, 20, rho = 1.0, gamma = gamma)
#        fs_sel_params.integrate(1, 20, rho = 1.0, sel_params = sel_params)
#        self.assertTrue(numpy.allclose(fs_gamma, fs_sel_params))
    
suite = unittest.TestLoader().loadTestsFromTestCase(TwoLocusTestCase)

if __name__ == '__main__':
    unittest.main()
