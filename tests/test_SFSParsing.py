import os
import unittest

import numpy as np
import moments
import pickle
import time


class TestParseSFS(unittest.TestCase):

    def test_defaults(self):
        # Execute with all default settings
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/basic_3_sample.vcf')  
        result = moments.Parsing.parse_vcf(vcf_file)
        expected = np.array([0, 2, 1, 1, 1, 0, 0])
        self.assertTrue(np.all(result == expected))

    def test_allow_multiallelic(self):
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/basic_3_sample.vcf')  
        result = moments.Parsing.parse_vcf(vcf_file, allow_multiallelic=True)
        expected = np.array([0, 2, 3, 1, 1, 0, 0])
        self.assertTrue(np.all(result == expected))

    def test_population_file(self):
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/basic_3_sample.vcf')  
        pop_file = os.path.join(os.path.dirname(__file__), 
            'test_files/3_pop_pop_file.txt')  
        result = moments.Parsing.parse_vcf(vcf_file, pop_file=pop_file)
        expected = np.array([
            [[0, 0, 0],
             [1, 0, 1],
             [0, 0, 0]],
            [[1, 0, 0],
             [1, 0, 1],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]]
        )
        self.assertTrue(np.all(result == expected))
        # Also test pop_mapping
        pop_mapping = {'A': ['sample1'], 'B': ['sample2'], 'C': ['sample3']}
        result = moments.Parsing.parse_vcf(vcf_file, pop_mapping=pop_mapping)
        self.assertTrue(np.all(result == expected))

    def test_ref_as_ancestral(self):
        # This is the default behavior.
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/multiallelic_2_sample.vcf')
        pop_mapping = {'A': ['sample1'], 'B': ['sample2']}
        
        result = moments.Parsing.parse_vcf(vcf_file, pop_mapping=pop_mapping)
        expected = np.array(
            [[0, 0, 1],
             [1, 1, 1],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

        # Allow multiallelic sites        
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, allow_multiallelic=True)
        expected = np.array(
            [[0, 1, 2],
             [3, 1, 1],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

    def test_use_ancestral_alleles(self):
        # Test using the INFO/AA field to polarize.
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/multiallelic_2_sample.vcf')
        
        result = moments.Parsing.parse_vcf(vcf_file, use_AA=True)
        expected = np.array([0, 1, 2, 0, 0])
        self.assertTrue(np.all(result == expected))

        # Allow multiallelic sites
        result = moments.Parsing.parse_vcf(
            vcf_file, use_AA=True, allow_multiallelic=True)
        expected = np.array([0, 5, 3, 1, 0])
        self.assertTrue(np.all(result == expected))

        # Multiple populations 
        pop_mapping = {'A': ['sample1'], 'B': ['sample2']}
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, use_AA=True)
        expected = np.array(
            [[0, 0, 1],
             [1, 1, 0],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))
        # Allow multiallelic sites
        result = moments.Parsing.parse_vcf(vcf_file, pop_mapping=pop_mapping, 
                                           use_AA=True, allow_multiallelic=True)
        expected = np.array(
            [[0, 1, 2],
             [4, 1, 1],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

    def test_fasta_ancestral_sequence(self):
        # Test proper behavior when using a FASTA file to assign anc. state.
        # Includes an example where the ancestral allele is unrepresented
        # among VCF alleles (position 5).
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/multiallelic_2_sample.vcf')
        fasta_file = os.path.join(os.path.dirname(__file__), 
            'test_files/fasta_file_full.fa')
        fasta_file_lowconf = os.path.join(os.path.dirname(__file__), 
            'test_files/fasta_file_low_conf.fa')
        pop_mapping = {'A': ['sample1'], 'B': ['sample2']}

        # Default
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, anc_seq_file=fasta_file
        )
        expected = np.array(
            [[0, 0, 1],
             [1, 1, 0],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

        # Allow multiallelic sites
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, anc_seq_file=fasta_file,
            allow_multiallelic=True
        )
        expected = np.array(
            [[0, 1, 2],
             [4, 1, 1],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

        # Use a sequence with low-confidence assignments
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, anc_seq_file=fasta_file_lowconf
        )
        expected = np.array(
            [[0, 0, 1],
             [0, 0, 0],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

        # Use a sequence with low-confidence assignments, accepting them
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, anc_seq_file=fasta_file_lowconf,
            allow_low_confidence=True
        )
        expected = np.array(
            [[0, 0, 1],
             [1, 1, 0],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

        # Also, allow multiallelic sites
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping, anc_seq_file=fasta_file_lowconf,
            allow_low_confidence=True, allow_multiallelic=True
        )
        expected = np.array(
            [[0, 1, 2],
             [4, 1, 1],
             [0, 0, 0]]
        )
        self.assertTrue(np.all(result == expected))

    def test_line_level_quality_filtering(self):
        # Tests numeric filters on DP and GQ in INFO, on QUAL and on FILTER
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/info_DP_GQ_3_sample.vcf')
        filters = {'INFO/DP': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 0, 1, 0, 1, 0])
        self.assertTrue(np.all(result == expected))
        filters = {'INFO/GQ': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 1, 0, 0, 1, 0])
        self.assertTrue(np.all(result == expected))
        filters = {'INFO/GQ': 30, 'INFO/DP': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 0, 0, 0, 1, 0])
        self.assertTrue(np.all(result == expected))
        filters = {'QUAL': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 1, 1, 0, 0, 0])
        self.assertTrue(np.all(result == expected))
        filters = {'FILTER': 'PASS'}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 1, 1, 0, 1, 0])
        self.assertTrue(np.all(result == expected))
        filters = {'FILTER': 'FAIL'}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 0, 0, 0, 0, 1, 0])
        self.assertTrue(np.all(result == expected))

    def test_sample_level_quality_filtering(self):
        # Because we don't provide `sample_sizes`, lines with filtered samples
        # are not included in the output.
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/format_DP_GQ_4_sample.vcf')
        
        result = moments.Parsing.parse_vcf(vcf_file)
        expected = np.array([0, 1, 1, 1, 1, 1, 2, 1, 0])
        self.assertTrue(np.all(result == expected))
        
        filters = {'SAMPLE/DP': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(result == expected))

        filters = {'SAMPLE/GQ': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(result == expected))

        filters = {'SAMPLE/DP': 30, 'SAMPLE/GQ': 30}
        result = moments.Parsing.parse_vcf(vcf_file, filters=filters)
        expected = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(result == expected))

    def test_sample_level_quality_filtering_projected(self):
        # As above, but we project to a smaller sample size also.
        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/format_DP_GQ_4_sample.vcf')
        
        filters = {'SAMPLE/DP': 30}
        result = moments.Parsing.parse_vcf(
            vcf_file, filters=filters, sample_sizes={'ALL': 6})
        expected = (
            moments.Spectrum(np.array([0, 1, 0, 1, 0, 0, 0, 0, 0])).project([6])
            + np.array([0, 0, 0, 1, 1, 0, 0])
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

        filters = {'SAMPLE/DP': 30, 'SAMPLE/GQ': 30}
        result = moments.Parsing.parse_vcf(
            vcf_file, filters=filters, sample_sizes={'ALL': 6})
        expected = (
            moments.Spectrum(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])).project([6])
            + np.array([0, 0, 0, 1, 1, 0, 0])
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_missing_data_projection(self):

        vcf_file = os.path.join(os.path.dirname(__file__), 
            'test_files/missing_data_2_sample.vcf')
        pop_mapping1 = {'AB': ['sample1', 'sample2']}
        pop_mapping2 = {'A': ['sample1'], 'B': ['sample2']}

        # Defaults
        result = moments.Parsing.parse_vcf(vcf_file)
        expected = np.array([0, 0, 0, 2, 0])
        self.assertTrue(np.all(result == expected))
        result = moments.Parsing.parse_vcf(vcf_file, pop_mapping=pop_mapping2)
        expected = np.array(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0]]
        )
        self.assertTrue(np.all(result == expected))

        # One-population projection
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping1, sample_sizes={'AB': 3})
        expected = np.array([0, 1, 1.5, 0])
        self.assertTrue(np.all(np.isclose(result, expected)))
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping1, sample_sizes={'AB': 2})
        expected = np.array([0, 5/3 + 2, 0])
        self.assertTrue(np.all(np.isclose(result, expected)))
        
        # Two-population projection
        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping2, sample_sizes={'A': 2, 'B': 1})
        expected = np.array(
            [[0, 0],
             [0, 1],
             [0.5, 0]]
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

        result = moments.Parsing.parse_vcf(
            vcf_file, pop_mapping=pop_mapping2, sample_sizes={'A': 1, 'B': 2})
        expected = np.array(
            [[0, 1, 0.5],
             [0, 1, 0]],
        )
        self.assertTrue(np.all(np.isclose(result, expected)))


class TestTallyVCF(unittest.TestCase):
    
    def test_minimal(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        expected = {(6,): {(4,): 1, (3,): 1, (2,): 1, (1,): 2}}
        result = moments.Parsing._tally_vcf(vcf_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_pop_file(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/3_pop_pop_file.txt'
        )
        expected = {(2, 2, 2): 
            {(1, 1, 2): 1, (0, 1, 2): 1, (1, 1, 0): 1,(0, 1, 0): 1,(1, 0, 0): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_pop_file_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/3_pop_pop_file.txt'
        )
        expected = {
            (2, 2, 2): {(1, 1, 2): 1, (0, 1, 2): 1, (1, 1, 0): 2, (0, 1, 0): 1,
                        (1, 0, 0): 1, (1, 0, 1): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file, allow_multiallelic=True)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_use_AA(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        expected = {(6,): {(4,): 2, (3,): 1, (1,): 2}}
        result = moments.Parsing._tally_vcf(vcf_file, use_AA=True)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_use_AA_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        expected = {(6,): {(4,): 2, (3,): 1, (1,): 2, (2,): 2}}
        result = moments.Parsing._tally_vcf(
            vcf_file, use_AA=True, allow_multiallelic=True
        )['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_anc_seq_file(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        fasta_file = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_full.fa'
        )
        expected = {(6,): {(4,): 2, (3,): 1, (1,): 2}}
        result = moments.Parsing._tally_vcf(
            vcf_file, anc_seq_file=fasta_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_anc_seq_file_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        fasta_file = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_full.fa'
        )
        expected = {(6,): {(4,): 2, (3,): 1, (1,): 2, (2,): 2}}
        result = moments.Parsing._tally_vcf(
            vcf_file, anc_seq_file=fasta_file, allow_multiallelic=True)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_missing_data(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/missing_data_3_sample.vcf'
        )

        expected = {
            (4,): {(2,): 1},
            (2,): {(2,): 1},
            (6,): {(4,): 1, (2,): 1},
            (3,): {(1,): 1}
        }
        result = moments.Parsing._tally_vcf(vcf_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_missing_data_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/missing_data_3_sample.vcf'
        )
        expected = {
            (4,): {(2,): 1, (1,): 2},
            (2,): {(2,): 1},
            (6,): {(4,): 1, (2,): 1},
            (3,): {(1,): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, allow_multiallelic=True)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_missing_data_with_pop_file(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/missing_data_3_sample.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/3_pop_pop_file.txt'
        )
        expected = {
            (2, 2, 0): {(1, 1, 0): 1},
            (0, 0, 2): {(0, 0, 2): 1},
            (2, 2, 2): {(2, 1, 1): 1, (0, 1, 1): 1}, 
            (1, 2, 0): {(0, 1, 0): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_missing_data_with_pop_file_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/missing_data_3_sample.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/3_pop_pop_file.txt'
        )
        expected = {
            (2, 2, 0): {(1, 1, 0): 1, (1, 0, 0): 1, (0, 1, 0): 1},
            (0, 0, 2): {(0, 0, 2): 1},
            (2, 2, 2): {(2, 1, 1): 1, (0, 1, 1): 1}, 
            (1, 2, 0): {(0, 1, 0): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file, allow_multiallelic=True)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_masking(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )        
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed'
        )
        expected = {(6,): {(4,): 1, (2,): 1, (1,): 1}} 
        result = moments.Parsing._tally_vcf(
            vcf_file, bed_file=bed_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_interval(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        expected = {(6,): {(4,): 1, (3,): 1, (2,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, interval=(0,4))['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_masking_and_interval(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )     
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed'
        )
        expected = {(6,): {(4,): 1, (2,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, interval=(0,4), bed_file=bed_file)['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_qual_filter(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf'
        )
        expected = {(6,): {(4,): 1, (3,): 1, (2,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, filters={'QUAL': 30})['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])

    def test_filter_filter(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/basic_3_sample.vcf')
        expected = {(6,): {(4,): 1, (3,): 1, (1,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, filters={'FILTER': 'PASS'})['tally']
        for ns in expected:
            for ms in expected[ns]:
                self.assertEqual(result[ns][ms], expected[ns][ms])


class TestSpectrumFromTally(unittest.TestCase):
    ## Test SFS construction from tallies of derived allele counts

    def setUp(self):
        self.data_1_pop = {
            'pop_ids': ['A'], 
            'sample_sizes': {'A': 4}, 
            'tally': {
                (3,): {(3,): 1, (2,): 3, (1,): 1},
                (4,): {(4,): 1, (3,): 5, (2,): 7, (1,): 10, (0,): 5}
            }
        }
        self.data_2_pop = {
            'pop_ids': ['A', 'B'],
            'sample_sizes': {'A': 4, 'B': 4},
            'tally': {
                (3, 3): {(2, 2): 10},
                (3, 4): {(2, 3): 10},
                (4, 3): {(3, 2): 10},
                (4, 4): {(3, 3): 10, (2, 2): 5}
            }
        }

    def tearDown(self):
        del(self.data_1_pop)
        del(self.data_2_pop)
    
    def test_one_pop_default(self):
        result = moments.Parsing._spectrum_from_tally(self.data_1_pop)
        expected = np.ma.array([0, 10, 7, 5, 0], mask=(1, 0, 0, 0, 1))
        self.assertTrue(np.all(result == expected))
        result = moments.Parsing._spectrum_from_tally(
            self.data_1_pop, mask_corners=False)
        expected = np.array([5, 10, 7, 5, 1])
        self.assertTrue(np.all(result == expected))

    def test_one_pop_projection(self):
        result = moments.Parsing._spectrum_from_tally(
            self.data_1_pop, sample_sizes={'A': 3}
        )
        expected = np.ma.array([0, 12, 10.25, 0], mask=(1, 0, 0, 1))
        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_two_pop_default(self):
        result = moments.Parsing._spectrum_from_tally(self.data_2_pop)
        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 5, 0, 0],
             [0, 0, 0, 10, 0],
             [0, 0, 0, 0, 0]]
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

    def test_two_pop_projection(self):
        result = moments.Parsing._spectrum_from_tally(
            self.data_2_pop, sample_sizes={'A': 3, 'B': 3})
        expected = np.array(
            [[0, 0, 0, 0],
             [0, 1.25, 1.25, 0],
             [0, 1.25, 31.875, 4.375],
             [0, 0, 4.375, 0]],
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

        result = moments.Parsing._spectrum_from_tally(
            self.data_2_pop, sample_sizes={'A': 4, 'B': 3})
        expected = np.array(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 2.5, 2.5, 0],
             [0, 0, 17.5, 2.5],
             [0, 0, 0, 0]],
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

        result = moments.Parsing._spectrum_from_tally(
            self.data_2_pop, sample_sizes={'A': 3, 'B': 4})
        expected = np.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 2.5, 0, 0],
             [0, 0, 2.5, 17.5, 0],
             [0, 0, 0, 2.5, 0]],
        )
        self.assertTrue(np.all(np.isclose(result, expected)))

class TestParseFilters(unittest.TestCase):
    
    def test_empty_filters(self):
        result = moments.Parsing._parse_filters(None)
        self.assertEqual(result, {})
        result = moments.Parsing._parse_filters({})
        self.assertEqual(result, {})

    def test_many_filters(self):
        # Ensure that types are handled properly
        input_dict = {
            'FILTER': 'PASS',
            'QUAL': 30,
            'INFO/CATEGORY': 'CATEGORY',
            'INFO/GQ': 30,
            'SAMPLE/GQ': 30,
            'SAMPLE/DP': 45,
            'FORMAT/CATEGORY': 'CATEGORY'
        }
        result = moments.Parsing._parse_filters(input_dict)
        expected = {
            'FILTER': set(['PASS']),
            'QUAL': 30.0,
            'INFO': {'GQ': 30.0, 'CATEGORY': set(['CATEGORY'])},
            'SAMPLE': {'GQ': 30.0, 'DP': 45.0, 'CATEGORY': set(['CATEGORY'])}
        }
        self.assertEqual(result, expected)

    def test_exceptions(self):
        # Test some invalid filter configurations.
        with self.assertRaises(ValueError):
            moments.Parsing._parse_filters({'INVALIDFIELD': 30})
        with self.assertRaises(ValueError):
            moments.Parsing._parse_filters({'INVALIDFIELD/GQ': 30})
        with self.assertRaises(ValueError):
            moments.Parsing._parse_filters({'INVALIDFIELD/GQ/GQ': 30})
        # Note that we don't check this sort of thing:
        input_filters = {'INFO/INVALIDFIELD': 30}
        result = moments.Parsing._parse_filters(input_filters)
        expected = {'INFO': {'INVALIDFIELD': 30.0}}
        self.assertEqual(result, expected)
        # Typing of FILTER
        with self.assertRaises(TypeError):
            moments.Parsing._parse_filters({'FILTER': 30})
        # Typing of QUAL
        with self.assertRaises(TypeError):
            moments.Parsing._parse_filters({'QUAL': 'X'})
        with self.assertRaises(TypeError):
            moments.Parsing._parse_filters({'QUAL': set(['X'])})


class TestFilters(unittest.TestCase):
    ## Tests of filtering functions. `True` means that a line fails a filter.
    
    def test_check_anc(self):
        # Invalid/missing AA triggers line skipping
        result = moments.Parsing._check_anc('A', False)
        expected = (False, 'A')
        self.assertEqual(result, expected)
        result = moments.Parsing._check_anc('a', False)
        expected = (True, None)
        self.assertEqual(result, expected)
        result = moments.Parsing._check_anc('a', True)
        expected = (False, 'A')
        self.assertEqual(result, expected)
        with self.assertWarns(UserWarning):
            result = moments.Parsing._check_anc('.', False)
            expected = (True, None)
            self.assertEqual(result, expected)
        with self.assertWarns(UserWarning):
            result = moments.Parsing._check_anc('N', False)
            expected = (True, None)
            self.assertEqual(result, expected)

    def test_filter_QUAL(self):
        # Invalid/missing QUAL does not trigger line skip
        self.assertFalse(moments.Parsing._filter_qual('31', 30))
        self.assertTrue(moments.Parsing._filter_qual('29', 30))
        self.assertFalse(moments.Parsing._filter_qual('.', 30))
        self.assertFalse(moments.Parsing._filter_qual('X', 30))
     
    def test_filter_FILTER(self):
        # Invalid/missing FILTER does not trigger line skip
        set1 = set(['PASS'])
        set2 = set(['PASS', 'FAIL'])
        self.assertFalse(moments.Parsing._filter_filter('PASS', set1))
        self.assertFalse(moments.Parsing._filter_filter('PASS', set2))
        self.assertTrue(moments.Parsing._filter_filter('FAIL', set1))
        self.assertFalse(moments.Parsing._filter_filter('.', set1))

    def test_filter_INFO(self):
        info = {'GQ': '30', 'DP': '30', 'X': '.'}
        # Absent INFO triggers a warning but does not skip a line
        with self.assertWarns(UserWarning):
            fltr = {'Y': 10}
            self.assertFalse(moments.Parsing._filter_info(info, fltr)) 
        fltr = {'GQ': 29.0}
        self.assertFalse(moments.Parsing._filter_info(info, fltr))
        fltr = {'DP': 29.0, 'GQ': 29.0}
        self.assertFalse(moments.Parsing._filter_info(info, fltr))
        fltr = {'DP': 29.0}
        self.assertFalse(moments.Parsing._filter_info(info, fltr))
        fltr = {'DP': 31.0, 'GQ': 31.0}
        self.assertTrue(moments.Parsing._filter_info(info, fltr))
        fltr = {'X': 10.0}
        self.assertFalse(moments.Parsing._filter_info(info, fltr))
        fltr = {'DP': 29.0, 'GQ': 29.0, 'X': 10.0}
        self.assertFalse(moments.Parsing._filter_info(info, fltr))
        fltr = {'DP': 29.0, 'GQ': 31.0, 'X': 10.0}
        self.assertTrue(moments.Parsing._filter_info(info, fltr))

    def test_filter_sample(self):
        sample = {'GT': '0|0', 'GQ': '30', 'DP': '30'}
        self.assertTrue(moments.Parsing._filter_sample(sample, {'GQ': 31.0}))
        self.assertTrue(moments.Parsing._filter_sample(sample, {'DP': 31.0}))
        self.assertTrue(moments.Parsing._filter_sample(
            sample, {'GQ': 31.0, 'DP': 31.0}))
        self.assertFalse(moments.Parsing._filter_sample(sample, {'GQ': 29.0}))
        self.assertFalse(moments.Parsing._filter_sample(sample, {'DP': 29.0}))
        self.assertFalse(moments.Parsing._filter_sample(
            sample, {'GQ': 29.0, 'DP': 29.0}))
        with self.assertWarns(UserWarning):
            self.assertFalse(moments.Parsing._filter_sample(sample, {'X': 10}))
        msample = {'GT': '0|0', 'GQ': '.', 'DP': '29'}
        self.assertFalse(moments.Parsing._filter_sample(msample, {'GQ': 30.0}))
        self.assertTrue(moments.Parsing._filter_sample(msample, {'DP': 30.0}))
        self.assertTrue(moments.Parsing._filter_sample(
            msample, {'GQ': 30.0, 'DP': 30.0}))
        

class TestComputeL(unittest.TestCase):
    
    def test_just_bed_file(self):
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        result = moments.Parsing.compute_L(bed_file)
        self.assertEqual(result, 6)

    def test_just_bed_file_sparse(self):
        bed_file_sparse = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        result = moments.Parsing.compute_L(bed_file_sparse)
        self.assertEqual(result, 3)
        
    def test_basic_anc_seq(self):
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        sparse_bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        anc_seq = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_full.fa')
        result = moments.Parsing.compute_L(bed_file, anc_seq_file=anc_seq)
        self.assertEqual(result, 6)
        result = moments.Parsing.compute_L(sparse_bed_file, 
            anc_seq_file=anc_seq)
        self.assertEqual(result, 3)

    def test_low_confidence_anc_seq(self):
        # Test operations on an ancestral sequence with low confidence states
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        sparse_bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        anc_seq = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_low_conf.fa')
        result = moments.Parsing.compute_L(bed_file, anc_seq_file=anc_seq)
        self.assertEqual(result, 2)
        result = moments.Parsing.compute_L(
            bed_file, anc_seq_file=anc_seq, allow_low_confidence=True)
        self.assertEqual(result, 6)
        # Sparse bed file
        result = moments.Parsing.compute_L(
            sparse_bed_file, anc_seq_file=anc_seq)
        self.assertEqual(result, 0)
        result = moments.Parsing.compute_L(
            sparse_bed_file, anc_seq_file=anc_seq, allow_low_confidence=True)
        self.assertEqual(result, 3)

    def test_missing_data_anc_seq(self):
        # Test with an ancestral sequence that includes missing data
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        sparse_bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        anc_seq = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_missing_data.fa')
        result = moments.Parsing.compute_L(bed_file, anc_seq_file=anc_seq)
        self.assertEqual(result, 2)
        result = moments.Parsing.compute_L(
            bed_file, anc_seq_file=anc_seq, allow_low_confidence=True)
        self.assertEqual(result, 3)
        result = moments.Parsing.compute_L(
            sparse_bed_file, anc_seq_file=anc_seq)
        self.assertEqual(result, 0)
        result = moments.Parsing.compute_L(
            sparse_bed_file, anc_seq_file=anc_seq, allow_low_confidence=True)
        self.assertEqual(result, 1)
        
    def test_intervaling(self):
        # Test intervaling.
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        bed_file_sparse = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        result = moments.Parsing.compute_L(bed_file, interval=(1, 4))
        self.assertEqual(result, 3)
        result = moments.Parsing.compute_L(bed_file, interval=(4, 7))
        self.assertEqual(result, 3)
        result = moments.Parsing.compute_L(bed_file, interval=(4, 666))
        self.assertEqual(result, 3)
        result = moments.Parsing.compute_L(bed_file_sparse, interval=(1, 4))
        self.assertEqual(result, 2)
        result = moments.Parsing.compute_L(bed_file_sparse, interval=(4, 6))
        self.assertEqual(result, 1)
        # Intervals are 1-indexed so beginning with 0 raises an error
        with self.assertRaises(ValueError):
            moments.Parsing.compute_L(bed_file, interval=(0, 6))


class TestLoadFASTAFile(unittest.TestCase):
    ## Test proper loading of FASTA files.
    
    def test_loading_multi_line_fasta(self):
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/fasta_file_large.fa')
        sequence = moments.Parsing._load_fasta_file(file_path)
        self.assertEqual(sequence, 'A' * 40)

    def test_loading_fasta_file(self):
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/fasta_file_missing_data.fa')
        sequence = moments.Parsing._load_fasta_file(file_path)
        self.assertEqual(sequence, '.NNAaA')  


class TestPopFile(unittest.TestCase):
    ## Test proper loading of population files.
    
    def test_loading_pop_file(self):
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/3_pop_pop_file.txt')
        mapping = moments.Parsing._load_pop_file(file_path)
        expected = {'A': ['sample1'], 'B': ['sample2'], 'C': ['sample3']}
        self.assertEqual(mapping, expected)

    def test_invalid_pop_file(self):
        # Should raise error when samples map to multiple populations
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/invalid_pop_file.txt')
        with self.assertRaises(ValueError):
            moments.Parsing._load_pop_file(file_path)


class TestLoadBEDFile(unittest.TestCase):
    ## Test proper loading of BED files.

    def test_loading_bed_file(self):
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/bed_file_full.bed')
        regions, chrom = moments.Parsing._load_bed_file(file_path)
        expected = np.array([[0, 6]])
        self.assertEqual(chrom, 'chr0')
        self.assertTrue(np.all(regions == expected))

    def test_loading_sparser_bed_file(self): 
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/bed_file_sparse.bed')
        regions, _ = moments.Parsing._load_bed_file(file_path)
        expected = np.array([[0, 1], [2, 3], [4, 5]])
        self.assertTrue(np.all(regions == expected))

    def test_scientific_format(self):
        # reading BED files with entries in scientific format e.g. 1.07e+08
        file_path = os.path.join(os.path.dirname(__file__), 
            'test_files/sci_format.bed')
        result, _ = moments.Parsing._load_bed_file(file_path)
        expected = np.array([[10500, 7600000], [95000000, 107000000]])
        self.assertTrue(np.all(result == expected))


class TestBEDRegionsToMask(unittest.TestCase):   
    ## Test handling of BED file regions.
    
    def test_regions_to_mask_continuous(self):
        regions = np.array([[0, 10]])
        result = moments.Parsing._bed_regions_to_mask(regions)
        self.assertTrue(np.all(result == False))
        self.assertTrue(len(result) == 10)

    def test_regions_to_mask_sparse(self):
        regions = np.array([[0, 1], [2, 3], [4, 5]])
        result = moments.Parsing._bed_regions_to_mask(regions)
        expected = np.array([False, True, False, True, False])
        self.assertTrue(np.all(result == expected))

    def test_regions_to_mask_single_site(self):
        regions = np.array([[4, 5]])
        result = moments.Parsing._bed_regions_to_mask(regions)
        expected = np.array([True, True, True, True, False])
        self.assertTrue(np.all(result == expected))

