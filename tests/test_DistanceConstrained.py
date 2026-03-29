import numpy as np
import unittest
import moments.LD

class Test_simple_constrianed_genotype_count(unittest.TestCase):
    L = 5
    n = 6
    np.random.seed(5553)
    G = np.random.randint(3, size=L * n).reshape(L, n)
    # working as the previous unmodified version
    D2_pw, Dz_pw, pi2_pw, D_pw = moments.LD.Parsing.compute_pairwise_stats(G, genotypes = True)

    def test_catch_pos_array_dtype(self):
        with self.assertRaises(AssertionError):
            pos_array = np.array([1, 2, 3, 4, 5], dtype = np.int64)
            threshold = 2
            moments.LD.Parsing.compute_pairwise_stats(self.G, genotypes = True,pos_array = pos_array, distance_constrained = threshold)

    def test_catch_pos_array_len_diff(self):
        with self.assertRaises(AssertionError):
            pos_array = np.array([1, 2, 3, 4], dtype = np.int32)
            threshold = 2
            moments.LD.Parsing.compute_pairwise_stats(self.G, genotypes = True,pos_array = pos_array, distance_constrained = threshold)

    def test_none_filtered(self):
        pos_array = np.array([1, 3, 5, 7, 9], dtype = np.int32)
        threshold = 1
        D2_pw, Dz_pw, pi2_pw, D_pw = moments.LD.Parsing.compute_pairwise_stats(self.G, genotypes = True, pos_array = pos_array, distance_constrained = threshold)
        self.assertTrue(np.all(D2_pw == self.D2_pw))
        self.assertTrue(np.all(Dz_pw == self.Dz_pw))
        self.assertTrue(np.all(D_pw == self.D_pw))
        self.assertTrue(np.all(pi2_pw == self.pi2_pw))
    
    def test_not_constrained(self):
        pos_array = np.array([1, 2, 3, 4, 5], dtype = np.int32)
        threshold = 0
        D2_pw, Dz_pw, pi2_pw, D_pw = moments.LD.Parsing.compute_pairwise_stats(self.G,  genotypes = True, pos_array = pos_array, distance_constrained = threshold)
        self.assertTrue(np.all(D2_pw == self.D2_pw))
        self.assertTrue(np.all(Dz_pw == self.Dz_pw))
        self.assertTrue(np.all(D_pw == self.D_pw))
        self.assertTrue(np.all(pi2_pw == self.pi2_pw))
    
    def test_all_filtered(self):
        pos_array = np.array([1, 2, 3, 4, 5], dtype = np.int32)
        threshold = 10
        D2_pw, Dz_pw, pi2_pw, D_pw = moments.LD.Parsing.compute_pairwise_stats(self.G, genotypes = True, pos_array = pos_array, distance_constrained = threshold)
        self.assertTrue(np.all(D2_pw == None))
        self.assertTrue(np.all(Dz_pw == None))
        self.assertTrue(np.all(D_pw == None))
        self.assertTrue(np.all(pi2_pw == None))
        

            
if __name__ == '__main__':
    unittest.main()