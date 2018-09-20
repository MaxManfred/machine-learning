import sys
import unittest
from unittest import makeSuite

from test.ml.classification.adaline.adaline_test import AdalineTest
from test.ml.classification.decision_tree.scikit_learn_decision_tree_test import ScikitLearnDecisionTreeTest
from test.ml.classification.knn.scikit_learn_knn_test import ScikitLearnKNNTest
from test.ml.classification.logistic_regression.logistic_regression_test import LogisticRegressionTest
from test.ml.classification.logistic_regression.scikit_learn_logistic_regression_test import \
    ScikitLearnLogisticRegressionTest
from test.ml.classification.perceptron.perceptron_test import PerceptronTest
from test.ml.classification.perceptron.scikit_learn_perceptron_test import ScikitLearnPerceptronTest
from test.ml.classification.svm.scikit_learn_svm_test import ScikitLearnSVMTest
from test.ml.data_preparation.categorical_data_test import CategoricalDataTest
from test.ml.data_preparation.missing_data_test import MissingDataTest
from test.ml.data_preparation.normalization_and_standardization_test import NormalizationStandardardizationTest
from test.ml.data_preparation.train_test_set_splitting_test import TrainTestSplittingTest


def suite():
    test_suite = unittest.TestSuite()

    # classification tests
    test_suite.addTest(makeSuite(AdalineTest))

    test_suite.addTest(makeSuite(ScikitLearnDecisionTreeTest))

    test_suite.addTest(makeSuite(ScikitLearnKNNTest))

    test_suite.addTest(makeSuite(LogisticRegressionTest))
    test_suite.addTest(makeSuite(ScikitLearnLogisticRegressionTest))

    test_suite.addTest(makeSuite(PerceptronTest))
    test_suite.addTest(makeSuite(ScikitLearnPerceptronTest))

    test_suite.addTest(makeSuite(ScikitLearnSVMTest))

    # data preparation tests
    test_suite.addTest(makeSuite(MissingDataTest))
    test_suite.addTest(makeSuite(CategoricalDataTest))
    test_suite.addTest(makeSuite(TrainTestSplittingTest))
    test_suite.addTest(makeSuite(NormalizationStandardardizationTest))

    return test_suite


mySuit = suite()

runner = unittest.TextTestRunner()
runner.run(mySuit)

sys.exit()
