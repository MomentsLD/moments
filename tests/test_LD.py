import os
import unittest

import numpy
import scipy.special
import moments
import moments.LD
import moments.TwoLocus
import pickle
import time

class LDTestCase(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_steady_state_fs(self):
        theta = 0.001
        fs = moments.Demographics1D.snm([20]) * theta
        y = moments.LD.Demographics1D.snm(theta=theta)
        self.assertTrue(numpy.allclose(y[-1][0], fs.project([2])))
        y = y.split(1)
        fs = moments.Manips.split_1D_to_2D(fs, 10, 10)
        fs_proj = fs.project([1,1])
        self.assertTrue(numpy.allclose(y[-1][1], fs_proj[0,1]+fs_proj[1,0]))
    
    def test_migration_symmetric_2D(self):
        theta = 0.001
        fs = moments.Demographics1D.snm([30]) * theta
        y = moments.LD.Demographics1D.snm(theta=theta)
        m = 1.0
        T = 0.3
        y = y.split(1)
        y.integrate([1,1], T, m=[[0,m],[m,0]], theta=theta)
        fs = moments.Manips.split_1D_to_2D(fs, 15, 15)
        fs.integrate([1,1], T, m=[[0,m],[m,0]], theta=theta)
        fs_proj = fs.project([1,1])
        self.assertTrue(numpy.allclose(y[-1][1], fs_proj[0,1]+fs_proj[1,0], rtol=1e-3))

    def test_migration_asymmetric_2D(self):
        theta = 0.001
        fs = moments.Demographics1D.snm([60]) * theta
        y = moments.LD.Demographics1D.snm(theta=theta)
        m12 = 10.0
        m21 = 0.0
        T = 2.0
        y = y.split(1)
        y.integrate([1,1], T, m=[[0,m12],[m21,0]], theta=theta)
        fs = moments.Manips.split_1D_to_2D(fs, 30, 30)
        fs.integrate([1,1], T, m=[[0,m12],[m21,0]], theta=theta)
        fs_proj = fs.project([1,1])
        self.assertTrue(numpy.allclose(y[-1][1], fs_proj[0,1]+fs_proj[1,0], rtol=1e-3))
        fs_proj = fs.marginalize([1]).project([2])
        self.assertTrue(numpy.allclose(y[-1][0], fs_proj[1], rtol=1e-3))
        fs_proj = fs.marginalize([0]).project([2])
        self.assertTrue(numpy.allclose(y[-1][2], fs_proj[1], rtol=1e-3))
    
    def test_equilibrium_ld_tlfs_slow(self):
        theta = 1
        rhos = [0, 1, 10]
        y = moments.LD.Demographics1D.snm(theta=theta, rho=rhos)
        ns = 50
        for ii,rho in enumerate(rhos):
            F = moments.TwoLocus.Demographics.equilibrium(ns, rho=rho).project(4)
            self.assertTrue(numpy.allclose(y[ii], [F.D2(), F.Dz(), F.pi2()], rtol=2e-2))

suite = unittest.TestLoader().loadTestsFromTestCase(LDTestCase)
