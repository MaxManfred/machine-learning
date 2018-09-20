import numpy as np
from numpy import count_nonzero
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.wine_dataset_test import WineDatasetTest


class RegularizationTest(WineDatasetTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_L1_regularization_with_scikit': cls.load_wine_data_set,
            'test_L2_regularization_with_scikit': cls.load_wine_data_set,
            'test_L1_L2_regularization_sparsity': cls.load_wine_data_set,
            'test_variable_L1_regularization_strength': cls.load_wine_data_set
        }

    def test_L1_regularization_with_scikit(self):
        # first, split the entire dataset into the training and testing subsets just for pretending this is a real
        # training scenario
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # standardize data sets
        X_train_standardized = StandardScaler().fit_transform(X_train)
        X_test_standardized = StandardScaler().fit_transform(X_test)

        # create a logistic regression classifier with L1 regularization
        l1_regularized_classifier = LogisticRegression(penalty='l1', C=1.0)

        # train the classifier
        l1_regularized_classifier.fit(X_train_standardized, Y_train)

        # check accuracy is the same on training and test set
        train_score1 = l1_regularized_classifier.score(X_train_standardized, Y_train)
        test_score1 = l1_regularized_classifier.score(X_test_standardized, Y_test)
        assert train_score1 == test_score1

    def test_L2_regularization_with_scikit(self):
        # first, split the entire dataset into the training and testing subsets just for pretending this is a real
        # training scenario
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # standardize data sets
        X_train_standardized = StandardScaler().fit_transform(X_train)
        X_test_standardized = StandardScaler().fit_transform(X_test)

        # create a logistic regression classifier with L2 regularization
        l2_regularized_classifier = LogisticRegression(penalty='l2', C=1.0)

        # train the classifier
        l2_regularized_classifier.fit(X_train_standardized, Y_train)

        # check accuracy on test set is even better that accuracy on training set
        train_score2 = l2_regularized_classifier.score(X_train_standardized, Y_train)
        test_score2 = l2_regularized_classifier.score(X_test_standardized, Y_test)
        assert train_score2 < test_score2

    def test_L1_L2_regularization_sparsity(self):
        # first, split the entire dataset into the training and testing subsets just for pretending this is a real
        # training scenario
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # standardize data sets
        X_train_standardized = StandardScaler().fit_transform(X_train)

        # create a logistic regression classifier with L1 regularization
        l1_regularized_classifier = LogisticRegression(penalty='l1', C=1.0)

        # train the classifier
        l1_regularized_classifier.fit(X_train_standardized, Y_train)

        # create a logistic regression classifier with L2 regularization
        l2_regularized_classifier = LogisticRegression(penalty='l2', C=1.0)

        # train the classifier
        l2_regularized_classifier.fit(X_train_standardized, Y_train)

        # check the weights arrays of the two classifiers and see that L1's contains multiple zeros as L1 creates a
        # sparse solution, while L2 doesn't
        assert count_nonzero(l1_regularized_classifier.coef_) < count_nonzero(l2_regularized_classifier.coef_)

    def test_variable_L1_regularization_strength(self):
        # first, split the entire dataset into the training and testing subsets just for pretending this is a real
        # training scenario
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # standardize data sets
        X_train_standardized = StandardScaler().fit_transform(X_train)

        weights, params = [], []

        for c in np.arange(-4., 6.):
            lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
            lr.fit(X_train_standardized, Y_train)
            weights.append(lr.coef_[1])
            params.append(10**c)

        weights = np.array(weights)

        Plotter.plot_variable_feature_weights(
            weights, params, self.df_columns,
            image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                'regularization/Logistic-Regression-Variable-L1-Regularized-Strength.png'
            )
        )
