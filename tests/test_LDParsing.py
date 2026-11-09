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


class ParsedDataNormalizingStatistics(unittest.TestCase):
    """
    bootstrap_data, get_bootstrap_sets, and subset_data keep the normalizing
    statistics in their outputs (since version 1.7.0).
    """

    def setUp(self):
        # Made-up statistic sums for 20 regions with two populations and two
        # recombination bins, formatted like the output of compute_ld_statistics.
        moment_names = moments.LD.Util.moment_names(2)
        self.stats = moment_names
        rng = np.random.default_rng(1)
        self.region_data = {
            i: {
                "bins": [(0, 1e-5), (1e-5, 1e-4)],
                "stats": moment_names,
                "pops": ["A", "B"],
                "sums": [
                    rng.random(len(moment_names[0])) + 1,
                    rng.random(len(moment_names[0])) + 1,
                    rng.random(len(moment_names[1])) + 1,
                ],
            }
            for i in range(20)
        }
        self.norm = [
            self.stats[0].index("pi2_0_0_0_0"),
            self.stats[1].index("H_0_0"),
        ]

    def test_bootstrap_data_and_sets(self):
        # Data means from bootstrap_data and every bootstrap set from
        # get_bootstrap_sets have all statistics, with the normalizing statistics
        # equal to 1 (and zero variance in the data).
        mv = moments.LD.Parsing.bootstrap_data(self.region_data)
        boots = moments.LD.Parsing.get_bootstrap_sets(self.region_data)
        self.assertEqual(len(boots), 20)
        for means in [mv["means"]] + boots:
            self.assertEqual([len(m) for m in means], [15, 15, 3])
            for m in means[:-1]:
                self.assertTrue(np.isclose(m[self.norm[0]], 1))
            self.assertTrue(np.isclose(means[-1][self.norm[1]], 1))
        self.assertTrue(np.isclose(mv["varcovs"][0][self.norm[0], self.norm[0]], 0))

    def test_get_bootstrap_sets_removed_options(self):
        # The remove_norm_stats and remove_Dz options were removed in 1.7.0.
        for option in ["remove_norm_stats", "remove_Dz"]:
            with self.assertRaises(TypeError):
                moments.LD.Parsing.get_bootstrap_sets(
                    self.region_data, **{option: True}
                )

    def test_subset_data(self):
        # Subsetting to population B (renamed population 0) keeps its statistics,
        # including pi2_0_0_0_0 and H_0_0, which are now B's normalizing
        # statistics. The returned statistics match the subset data.
        mv = moments.LD.Parsing.bootstrap_data(self.region_data)
        r_edges, means, varcovs, stats = moments.LD.Parsing.subset_data(mv, ["B"])
        self.assertEqual(stats, [list(s) for s in moments.LD.Util.moment_names(1)])
        self.assertEqual([len(m) for m in means], [len(stats[0])] * 2 + [1])
        self.assertEqual(varcovs[0].shape, (len(stats[0]), len(stats[0])))
        # all statistics when keeping both populations, in the same order
        _, means, _, stats = moments.LD.Parsing.subset_data(mv, ["A", "B"])
        self.assertEqual(stats, [list(s) for s in self.stats])
        self.assertTrue(np.allclose(means[0], mv["means"][0]))

    def test_subset_data_reorders_populations(self):
        # pops_to given out of order (["B", "A"]) warns, and the populations are
        # reordered to match the data, giving the same result as ["A", "B"], with
        # statistic names in the standard order.
        mv = moments.LD.Parsing.bootstrap_data(self.region_data)
        with self.assertWarns(UserWarning):
            out = moments.LD.Parsing.subset_data(mv, ["B", "A"])
        expected = moments.LD.Parsing.subset_data(mv, ["A", "B"])
        self.assertEqual(out[3], expected[3])
        self.assertEqual(out[3], [list(s) for s in self.stats])
        for a, b in zip(out[1], expected[1]):
            self.assertTrue(np.array_equal(a, b))

    def test_subset_without_normalizing_population(self):
        # Keeping only B, while the data are normalized by A, gives "normalizing"
        # statistics (B's pi2 and H, renamed to population 0) that don't equal 1,
        # so the inference functions raise an error instead of silently fitting
        # data and a model normalized by different populations.
        mv = moments.LD.Parsing.bootstrap_data(self.region_data)
        _, means, varcovs, stats = moments.LD.Parsing.subset_data(mv, ["B"])
        with self.assertRaises(ValueError) as err:
            moments.LD.Inference.remove_normalized_data(
                means, varcovs, num_pops=1, statistics=stats
            )
        self.assertIn("not 1", str(err.exception))

    def test_subset_data_keyword_options(self):
        # Options after pops_to must be given by keyword, so that calls written for
        # the removed normalization argument fail instead of shifting r_min/r_max.
        mv = moments.LD.Parsing.bootstrap_data(self.region_data)
        with self.assertRaises(TypeError):
            moments.LD.Parsing.subset_data(mv, ["A", "B"], 0, 1e-5)
        r_edges, means, _, _ = moments.LD.Parsing.subset_data(
            mv, ["A", "B"], r_max=1e-5
        )
        self.assertEqual(list(r_edges), [0, 1e-5])
        self.assertEqual(len(means), 2)


class ComputeLDStatisticsBins(unittest.TestCase):
    def test_r_bins_require_recombination_map(self):
        # r_bins are recombination distances (in Morgans), which can't be computed
        # without a recombination map. This used to silently compute no LD
        # statistics. Now it raises an error before reading the VCF (which doesn't
        # exist here), pointing to bp_bins for binning by physical distance.
        with self.assertRaises(ValueError) as err:
            moments.LD.Parsing.compute_ld_statistics(
                "not_a_file.vcf", r_bins=[0, 1e-5, 1e-4]
            )
        self.assertIn("bp_bins", str(err.exception))
