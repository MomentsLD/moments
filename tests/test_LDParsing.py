import os
import tempfile
import unittest

import numpy as np
import pytest
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


class ComputeLDStatisticsArguments(unittest.TestCase):
    """
    compute_ld_statistics checks its arguments before reading the VCF, which can be
    slow. Errors are tested by calling compute_ld_statistics with a VCF
    that doesn't exist, so that each test also confirms the check runs before the
    VCF is read (and before scikit-allel is needed). Warnings are tested on the
    checking function, since compute_ld_statistics would then go on to the VCF.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.vcf_file = os.path.join(self.tmpdir.name, "not_a_file.vcf")
        self.pop_file = os.path.join(self.tmpdir.name, "pops.txt")
        with open(self.pop_file, "w") as fout:
            fout.write("sample pop\ns1 A\ns2 A\ns3 B\ns4 B\n")
        self.map_file = os.path.join(self.tmpdir.name, "map.txt")
        with open(self.map_file, "w") as fout:
            fout.write("Position(bp) Map(cM) Other(cM)\n0 0 0\n1000000 1 2\n")
        self.bed_file = os.path.join(self.tmpdir.name, "regions.bed")
        with open(self.bed_file, "w") as fout:
            fout.write("1 0 1000000\n")

    def tearDown(self):
        self.tmpdir.cleanup()

    def compute(self, **kwargs):
        return moments.LD.Parsing.compute_ld_statistics(self.vcf_file, **kwargs)

    def check_args(self, **kwargs):
        args = dict(
            bed_file=None,
            region=None,
            rec_map_file=None,
            pop_file=None,
            pops=None,
            r_bins=None,
            bp_bins=[0, 1000, 10000],
            min_bp=None,
            stats_to_compute=None,
        )
        args.update(kwargs)
        moments.LD.Parsing._check_ld_statistics_args(**args)

    def test_valid_arguments(self):
        # Valid arguments pass the checks without error or warning (warnings are
        # errors in the tests), with r_bins and a map, and with bp_bins.
        self.check_args(
            bed_file=self.bed_file,
            region=[100, 200000],
            rec_map_file=self.map_file,
            pop_file=self.pop_file,
            pops=["B", "A"],
            r_bins=[0, 1e-5, 1e-4],
            bp_bins=None,
            min_bp=500,
            stats_to_compute=[["DD_0_0", "pi2_0_0_0_0"], ["H_0_0", "H_0_1"]],
        )
        self.check_args(pop_file=self.pop_file, pops=["A"])

    def test_r_bins_require_recombination_map(self):
        # r_bins are recombination distances (in Morgans), which can't be computed
        # without a recombination map.
        with self.assertRaises(ValueError) as err:
            self.compute(r_bins=[0, 1e-5, 1e-4])
        self.assertIn("bp_bins", str(err.exception))

    def test_r_bins_and_bp_bins(self):
        # Only one type of bins can be given.
        with self.assertRaises(ValueError):
            self.compute(
                rec_map_file=self.map_file, r_bins=[0, 1e-5], bp_bins=[0, 1000]
            )

    def test_invalid_bins(self):
        # Bins need at least two edges, which are non-negative and increasing, for
        # both types of bins.
        for bins in [[1000], [], [-100, 0, 100], [0, 1000, 1000], [0, 1000, 100]]:
            with self.assertRaises(ValueError):
                self.compute(bp_bins=bins)
            with self.assertRaises(ValueError):
                self.compute(rec_map_file=self.map_file, r_bins=bins)

    def test_missing_input_files(self):
        # The bed, map and population files must exist.
        missing = os.path.join(self.tmpdir.name, "missing.txt")
        for kwargs in [
            dict(bed_file=missing),
            dict(rec_map_file=missing, r_bins=[0, 1e-5]),
            dict(pop_file=missing, pops=["A"]),
        ]:
            with self.assertRaises(FileNotFoundError):
                self.compute(bp_bins=[0, 1000], **kwargs)

    def test_pops_and_pop_file_given_together(self):
        # pops and pop_file must be given together.
        with self.assertRaises(ValueError):
            self.compute(bp_bins=[0, 1000], pops=["A", "B"])
        with self.assertRaises(ValueError):
            self.compute(bp_bins=[0, 1000], pop_file=self.pop_file)

    def test_pops_not_in_pop_file(self):
        # Each of the pops must be in the population file, and the error names the
        # missing ones.
        with self.assertRaises(ValueError) as err:
            self.compute(bp_bins=[0, 1000], pop_file=self.pop_file, pops=["A", "C"])
        self.assertIn("C", str(err.exception))

    def test_pop_file_header(self):
        # The population file's first line names the sample and pop columns.
        no_header = os.path.join(self.tmpdir.name, "no_header.txt")
        with open(no_header, "w") as fout:
            fout.write("s1 A\ns2 B\n")
        with self.assertRaises(ValueError):
            self.compute(bp_bins=[0, 1000], pop_file=no_header, pops=["A"])

    def test_invalid_region(self):
        # A region is [start, end] with start < end.
        for region in [[100], [100, 200, 300], [200, 100], [100, 100]]:
            with self.assertRaises(ValueError):
                self.compute(bp_bins=[0, 1000], region=region)

    def test_invalid_stats_to_compute(self):
        # stats_to_compute is [ld_stats, h_stats], with names valid for the number
        # of populations (one, if pops aren't given).
        for stats in [
            ["DD_0_0", "pi2_0_0_0_0"],
            [["DD_0_0"], ["H_0_0"], []],
            [["DD_0_1"], ["H_0_0"]],
            [["DD_0_0"], ["H_0_1"]],
            [["H_0_0"], ["DD_0_0"]],
        ]:
            with self.assertRaises(ValueError):
                self.compute(bp_bins=[0, 1000], stats_to_compute=stats)
        # with two populations, statistics of a third population are invalid
        with self.assertRaises(ValueError):
            self.compute(
                bp_bins=[0, 1000],
                pop_file=self.pop_file,
                pops=["A", "B"],
                stats_to_compute=[["DD_0_2"], ["H_0_0"]],
            )

    def test_warn_no_bins(self):
        # Without bins, only heterozygosity statistics are computed.
        with self.assertWarns(UserWarning):
            self.check_args(bp_bins=None)

    def test_warn_bp_bins_with_map(self):
        # With bp_bins, a recombination map isn't used.
        with self.assertWarns(UserWarning):
            self.check_args(rec_map_file=self.map_file)

    def test_warn_min_bp_without_bed_file(self):
        # min_bp filters the features of a bed file, so it's ignored without one.
        with self.assertWarns(UserWarning):
            self.check_args(min_bp=1000)


class AssignRecombinationRates(unittest.TestCase):
    def setUp(self):
        pytest.importorskip("pandas")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.map_file = os.path.join(self.tmpdir.name, "map.txt")
        with open(self.map_file, "w") as fout:
            fout.write("Position(bp) Other(cM) Map(cM)\n0 0 0\n1000000 2 1\n")
        self.positions = np.array([250000, 500000])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_map_name(self):
        # The named map column is used, and Map(cM) if no map_name is given. Maps in
        # cM are converted to Morgans.
        rs = moments.LD.Parsing._assign_recombination_rates(
            self.positions, self.map_file, map_name="Other(cM)", report=False
        )
        self.assertTrue(np.allclose(rs, [0.005, 0.01]))
        rs = moments.LD.Parsing._assign_recombination_rates(
            self.positions, self.map_file, report=False
        )
        self.assertTrue(np.allclose(rs, [0.0025, 0.005]))

    def test_map_name_not_in_map_file(self):
        # A map_name that isn't in the map file warns, and
        # the first map column is used.
        with self.assertWarns(UserWarning):
            rs = moments.LD.Parsing._assign_recombination_rates(
                self.positions, self.map_file, map_name="Missing(cM)", report=False
            )
        self.assertTrue(np.allclose(rs, [0.005, 0.01]))
