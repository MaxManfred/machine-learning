import sys
import unittest
from unittest import makeSuite

from test.ml.classification.adaline.adaline_test import AdalineTest
from test.ml.classification.logistic_regression.logistic_regression_test import LogisticRegressionTest
from test.ml.classification.logistic_regression.scikit_learn_logistic_regression_test import \
    ScikitLearnLogisticRegressionTest
from test.ml.classification.perceptron.perceptron_test import PerceptronTest
from test.ml.classification.perceptron.scikit_learn_perceptron_test import ScikitLearnPerceptronTest
from test.ml.classification.svm.scikit_learn_svm_test import ScikitLearnSVMTest


def suite():
    test_suite = unittest.TestSuite()

    # classification tests
    test_suite.addTest(makeSuite(AdalineTest))

    test_suite.addTest(makeSuite(LogisticRegressionTest))
    test_suite.addTest(makeSuite(ScikitLearnLogisticRegressionTest))

    test_suite.addTest(makeSuite(PerceptronTest))
    test_suite.addTest(makeSuite(ScikitLearnPerceptronTest))

    test_suite.addTest(makeSuite(ScikitLearnSVMTest))

    return test_suite


mySuit = suite()

runner = unittest.TextTestRunner()
runner.run(mySuit)

sys.exit()
