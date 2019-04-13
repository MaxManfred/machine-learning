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
from test.ml.clustering.clustered_data_test import ClusteredDataTest
from test.ml.clustering.clustering_quality_test import ClusteringQualityTest
from test.ml.data_preparation.categorical_data_test import CategoricalDataTest
from test.ml.data_preparation.feature_importance_test import FeatureImportanceTest
from test.ml.data_preparation.missing_data_test import MissingDataTest
from test.ml.data_preparation.normalization_and_standardization_test import NormalizationStandardizationTest
from test.ml.data_preparation.sequential_feature_selection_test import SequentialFeatureSelectionTest
from test.ml.data_preparation.train_test_set_splitting_test import TrainTestSplittingTest
from test.ml.model_performance.scikit_learn_confusion_matrix_test import ScikitLearnConfusionMatrixTest
from test.ml.model_performance.scikit_learn_custom_scorer_test import ScikitLearnCustomScorerTest
from test.ml.model_performance.scikit_learn_grid_search_test import ScikitLearnGridSearchTest
from test.ml.model_performance.scikit_learn_k_fold_cv_test import ScikitLearnKFoldCVTest
from test.ml.model_performance.scikit_learn_macro_micro_averaging_test import ScikitLearnMacroMicroAveragingTest
from test.ml.model_performance.scikit_learn_nested_cv_test import ScikitLearnKNestedCVTest
from test.ml.model_performance.scikit_learn_performance_curves_test import ScikitLearnPerformanceCurvesTest
from test.ml.model_performance.scikit_learn_roc_auc_test import ScikitLearnROCAUCTest
from test.ml.pipeline.scikit_learn_pipeline_test import ScikitLearnPipelineTest
from test.ml.regularization.regularization_test import RegularizationTest


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
    test_suite.addTest(makeSuite(NormalizationStandardizationTest))
    test_suite.addTest(makeSuite(RegularizationTest))
    test_suite.addTest(makeSuite(SequentialFeatureSelectionTest))
    test_suite.addTest(makeSuite(FeatureImportanceTest))

    # model performance tests
    test_suite.addTest(makeSuite(ScikitLearnKFoldCVTest))
    test_suite.addTest(makeSuite(ScikitLearnKNestedCVTest))
    test_suite.addTest(makeSuite(ScikitLearnPerformanceCurvesTest))
    test_suite.addTest(makeSuite(ScikitLearnGridSearchTest))
    test_suite.addTest(makeSuite(ScikitLearnConfusionMatrixTest))
    test_suite.addTest(makeSuite(ScikitLearnCustomScorerTest))
    test_suite.addTest(makeSuite(ScikitLearnROCAUCTest))
    test_suite.addTest(makeSuite(ScikitLearnMacroMicroAveragingTest))

    # pipeline tests
    test_suite.addTest(makeSuite(ScikitLearnPipelineTest))

    # clustering analysis tests
    test_suite.addTest(makeSuite(ClusteredDataTest))
    test_suite.addTest(makeSuite(ClusteringQualityTest))

    return test_suite


mySuit = suite()

runner = unittest.TextTestRunner()
runner.run(mySuit)

sys.exit()
