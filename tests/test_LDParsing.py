import os
import unittest

import numpy as np
import moments.LD
import pickle
import time
import copy


class ValidGenotypeMatrices(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.3f seconds" % (self.id(), t))

    def test_invalid_genotype_matrix(self):
        genotypes = True
        G = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, 2], [0, 1, -2]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, 2], [0, 0, 1e-14]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, 2], [-1, 1, 2]])
        moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)

    def test_invalid_haplotype_matrix(self):
        genotypes = False
        G = np.array([[0, 1, 2], [3, 4, 5]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, 2], [0, 1, -2]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, 2], [0, 0, 1e-14]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, 2], [0, 1, 2]])
        with self.assertRaises(ValueError):
            moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)
        G = np.array([[0, 1, -1], [0, 1, -1]])
        moments.LD.Parsing._check_valid_genotype_matrix(G, genotypes)

    def test_missing_data(self):
        data1 = np.zeros([1, 10])
        data2 = np.zeros([1, 10])
        data1[0, 1] = -1
        data1[0, 4] = 1
        data2[0, 2] = 1
        stats = moments.LD.Parsing.compute_pairwise_stats_between(
            data1, data2, genotypes=True
        )

        data1 = np.zeros([1, 10])
        data2 = np.zeros([1, 10])
        data2[0, 1] = -1
        data2[0, 4] = 1
        data1[0, 2] = 1
        stats = moments.LD.Parsing.compute_pairwise_stats_between(
            data1, data2, genotypes=True
        )

