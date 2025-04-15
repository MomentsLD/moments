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
            os.path.dirname(__file__), 'test_files/vcf_file_basic.vcf')
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
    
    def test_check_anc(self):
        # `True` means that a line fails a filter
        self.assertFalse(moments.Parsing._check_anc('A', False))
        self.assertTrue(moments.Parsing._check_anc('a', False))
        self.assertFalse(moments.Parsing._check_anc('a', True))
        with self.assertWarns(UserWarning):
            moments.Parsing._check_anc('.', False)
        with self.assertWarns(UserWarning):
            moments.Parsing._check_anc('N', False)

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
            self.assertFalse(moments.Parsing._filter_info(info, {'Y': 10})) 
        self.assertFalse(moments.Parsing._filter_info(info, {'GQ': 29.0}))
        self.assertFalse(moments.Parsing._filter_info(info, 
            {'DP': 29.0, 'GQ': 29.0}))
        self.assertFalse(moments.Parsing._filter_info(info, {'DP': 29.0}))
        self.assertTrue(moments.Parsing._filter_info(info, 
            {'DP': 31.0, 'GQ': 31.0}))
        self.assertFalse(moments.Parsing._filter_info(info, {'X': 10.0}))
        self.assertFalse(moments.Parsing._filter_info(info, 
            {'DP': 29.0, 'GQ': 29.0, 'X': 10.0}))
        self.assertTrue(moments.Parsing._filter_info(info, 
            {'DP': 29.0, 'GQ': 31.0, 'X': 10.0}))

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
    
    def test_just_bedfile(self):
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        result = moments.Parsing.compute_L(bed_file)
        self.assertEqual(result, 6)
        bed_file_sparse = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        result = moments.Parsing.compute_L(bed_file_sparse)
        self.assertEqual(result, 3)
        
    def test_anc_seq(self):
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        full_seq = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_full.fa')
        low_conf_seq = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_low_conf.fa')
        missing_data_seq = os.path.join(
            os.path.dirname(__file__), 'test_files/fasta_file_missing_data.fa')
        result = moments.Parsing.compute_L(bed_file, anc_seq_file=full_seq)
        self.assertEqual(result, 6)
        result = moments.Parsing.compute_L(bed_file, anc_seq_file=low_conf_seq)
        self.assertEqual(result, 2)
        result = moments.Parsing.compute_L(
            bed_file, anc_seq_file=low_conf_seq, allow_low_confidence=True)
        self.assertEqual(result, 6)
        result = moments.Parsing.compute_L(
            bed_file, anc_seq_file=missing_data_seq)
        self.assertEqual(result, 2)
        result = moments.Parsing.compute_L(
            bed_file, anc_seq_file=missing_data_seq, allow_low_confidence=True)
        self.assertEqual(result, 3)
        
    def test_interval(self):
        bed_file = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_full.bed')
        bed_file_sparse = os.path.join(
            os.path.dirname(__file__), 'test_files/bed_file_sparse.bed')
        result = moments.Parsing.compute_L(bed_file, interval=(0, 3))
        self.assertEqual(result, 3)
        result = moments.Parsing.compute_L(bed_file, interval=(3, 6))
        self.assertEqual(result, 3)
        result = moments.Parsing.compute_L(bed_file_sparse, interval=(0, 3))
        self.assertEqual(result, 2)
        result = moments.Parsing.compute_L(bed_file_sparse, interval=(3, 5))
        self.assertEqual(result, 1)
        with self.assertWarns(UserWarning):
            moments.Parsing.compute_L(bed_file, interval=(0, 7))
        with self.assertWarns(UserWarning):
            moments.Parsing.compute_L(bed_file_sparse, interval=(3, 6))


class TestFASTAFiles(unittest.TestCase):
    pass


class TestPopFile(unittest.TestCase):
    pass


class TestBEDFiles(unittest.TestCase):
    pass