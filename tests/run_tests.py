from glob import glob
import unittest

import dadi

if __name__ == '__main__':
    # First we collect all our tests into a single TestSuite object.
    all_tests = unittest.TestSuite()

    testfiles = glob('test_*.py')
    all_test_mods = []
    for file in testfiles:
        module = file[:-3]
        mod = __import__(module)
        all_tests.addTest(mod.suite)
    unittest.TextTestRunner(verbosity=2).run(all_tests)
