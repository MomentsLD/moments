import os
import unittest

import numpy as np
import moments
import pickle
import time


class TestParseSFS(unittest.TestCase):
    pass


class TestTallyVCF(unittest.TestCase):
    pass


class TestSpectrumFromTally(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_one_pop_default(self):
        pass

    def test_one_pop_projection(self):
        pass

    def test_two_pop_default(self):
        pass

    def test_two_pop_projection(self):
        pass

    def test_masking(self):
        pass

    def test_pop_ids(self):
        pass

    def test_empty(self):
        pass


class TestParseFilters(unittest.TestCase):
    pass


class TestFilters(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_check_anc():
        pass

    def test_filter_QUAL():
        pass
     
    def test_filter_FILTER():
        pass

    def test_filter_INFO():
        pass

    def test_filter_sample():
        pass


class TestComputeL(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_just_bedfile(self):
        pass

    def test_anc_seq(self):
        pass


class TestFASTAFiles(unittest.TestCase):
    pass


class TestPopFile(unittest.TestCase):
    pass


class TestBEDFiles(unittest.TestCase):
    pass