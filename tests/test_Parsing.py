import os
import unittest

import numpy as np
import moments
import pickle
import time


class TestParseSFS(unittest.TestCase):
    pass


class TestTallyVCF(unittest.TestCase):
    
    def test_minimal(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        proper_result = {(6,): {(4,): 1, (3,): 1, (2,): 1, (1,): 2}}
        result = moments.Parsing._tally_vcf(vcf_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_pop_file(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/pop_file.txt'
        )
        proper_result = {
            (2, 2, 2): {(1, 1, 2): 1, (0, 1, 2): 1, (1, 1, 0): 1, (0, 1, 0): 1,
                        (1, 0, 0): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_pop_file_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/pop_file.txt'
        )
        proper_result = {
            (2, 2, 2): {(1, 1, 2): 1, (0, 1, 2): 1, (1, 1, 0): 2, (0, 1, 0): 1,
                        (1, 0, 0): 1, (1, 0, 1): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file, allow_multiallelic=True)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_use_AA(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        proper_result = {(6,): {(4,): 2, (3,): 1, (1,): 2}}
        result = moments.Parsing._tally_vcf(vcf_file, use_AA=True)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_use_AA_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        proper_result = {(6,): {(4,): 2, (3,): 1, (1,): 2, (2,): 2}}
        result = moments.Parsing._tally_vcf(
            vcf_file, use_AA=True, allow_multiallelic=True
        )['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_anc_seq_file(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        fasta_file = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_full.fa'
        )
        proper_result = {(6,): {(4,): 2, (3,): 1, (1,): 2}}
        result = moments.Parsing._tally_vcf(
            vcf_file, anc_seq_file=fasta_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_anc_seq_file_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        fasta_file = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_full.fa'
        )
        proper_result = {(6,): {(4,): 2, (3,): 1, (1,): 2, (2,): 2}}
        result = moments.Parsing._tally_vcf(
            vcf_file, anc_seq_file=fasta_file, allow_multiallelic=True)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_missing_data(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_missing_data.vcf'
        )

        proper_result = {
            (4,): {(2,): 1},
            (2,): {(2,): 1},
            (6,): {(4,): 1, (2,): 1},
            (3,): {(1,): 1}
        }
        result = moments.Parsing._tally_vcf(vcf_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_missing_data_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_missing_data.vcf'
        )
        proper_result = {
            (4,): {(2,): 1, (1,): 2},
            (2,): {(2,): 1},
            (6,): {(4,): 1, (2,): 1},
            (3,): {(1,): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, allow_multiallelic=True)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_missing_data_with_pop_file(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_missing_data.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/pop_file.txt'
        )
        proper_result = {
            (2, 2, 0): {(1, 1, 0): 1},
            (0, 0, 2): {(0, 0, 2): 1},
            (2, 2, 2): {(2, 1, 1): 1, (0, 1, 1): 1}, 
            (1, 2, 0): {(0, 1, 0): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_missing_data_with_pop_file_multiallelic(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_missing_data.vcf'
        )
        pop_file = os.path.join(
            os.path.dirname(__file__), 'test_files/pop_file.txt'
        )
        proper_result = {
            (2, 2, 0): {(1, 1, 0): 1, (1, 0, 0): 1, (0, 1, 0): 1},
            (0, 0, 2): {(0, 0, 2): 1},
            (2, 2, 2): {(2, 1, 1): 1, (0, 1, 1): 1}, 
            (1, 2, 0): {(0, 1, 0): 1}
        }
        result = moments.Parsing._tally_vcf(
            vcf_file, pop_file=pop_file, allow_multiallelic=True)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_masking(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )        
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed'
        )
        proper_result = {(6,): {(4,): 1, (2,): 1, (1,): 1}} 
        result = moments.Parsing._tally_vcf(
            vcf_file, bed_file=bed_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_interval(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        proper_result = {(6,): {(4,): 1, (3,): 1, (2,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, interval=(0,4))['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_masking_and_interval(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )     
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed'
        )
        proper_result = {(6,): {(4,): 1, (2,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, interval=(0,4), bed_file=bed_file)['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_qual_filter(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        proper_result = {(6,): {(4,): 1, (3,): 1, (2,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, filters={'QUAL': 30})['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])

    def test_filter_filter(self):
        vcf_file = os.path.join(
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf'
        )
        proper_result = {(6,): {(4,): 1, (3,): 1, (1,): 1}}
        result = moments.Parsing._tally_vcf(
            vcf_file, filters={'FILTER': 'PASS'})['tally']
        for ns in proper_result:
            for ms in proper_result[ns]:
                self.assertEqual(result[ns][ms], proper_result[ns][ms])


class TestSpectrumFromTally(unittest.TestCase):
    
    def test_one_pop_default(self):
        data = {'pop_ids': ['A'], 'sample_sizes': {'A': 4}, 
                'tally': {(4,): {(4,): 1, (3,): 3, (2,): 8, (1,): 12, (0,): 4}}}
        proper_result = np.array([0, 12, 8, 3, 0])
        result = moments.Parsing._spectrum_from_tally(data)
        self.assertTrue(np.all(result == proper_result))

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
    
    def test_check_anc(self):
        pass

    def test_filter_QUAL(self):
        pass
     
    def test_filter_FILTER(self):
        pass

    def test_filter_INFO(self):
        pass

    def test_filter_sample(self):
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