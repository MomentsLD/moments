import unittest
import numpy
import moments

class ProjectionTestCase(unittest.TestCase):
    def test_project_up(self):
        """
        Saving spectrum to file.
        """
        fixed_params = [0.1,None,None]
        params_up = moments.Inference._project_params_up([0.2,0.3], fixed_params)
        self.assertTrue(numpy.allclose(params_up, [0.1,0.2,0.3]))

        fixed_params = [0.1,0.2,None]
        params_up = moments.Inference._project_params_up([0.3], fixed_params)
        self.assertTrue(numpy.allclose(params_up, [0.1,0.2,0.3]))

        fixed_params = [0.1,0.2,None]
        params_up = moments.Inference._project_params_up(0.3, fixed_params)
        self.assertTrue(numpy.allclose(params_up, [0.1,0.2,0.3]))

suite = unittest.TestLoader().loadTestsFromTestCase(ProjectionTestCase)
if __name__ == '__main__':
    unittest.main()
