import unittest

from tests.adaline.adaline_test import AdalineTest


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(AdalineTest))

    return test_suite


mySuit = suite()

runner = unittest.TextTestRunner()
runner.run(mySuit)
