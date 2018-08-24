import unittest

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from ml.classification.logistic_regression.logistic_regression import LogisticRegressionBGD
from ml.common.plot.plotter import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils


class LogisticRegressionTest(unittest.TestCase):

    def setUp(self):
        # load subset of Iris data
        iris = datasets.load_iris()
        x_train = iris.data[:, [2, 3]]
        y_train = iris.target

        # consider only 0 and 1 labels
        x_train_01_subset = x_train[(y_train == 0) | (y_train == 1)]
        y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

        # Standardize features
        sc = StandardScaler()
        sc.fit(x_train_01_subset)
        self.x = sc.transform(x_train_01_subset)

        self.y = y_train_01_subset
        print('Class labels:', np.unique(self.y))

        # plotter data and save it to file
        Plotter.plot_iris_data_set(self.x, FilesystemUtils.get_test_resources_plot_file_name(
            'logistic_regression/LogisticRegressionBGD-Training-Set.png'))

    def tearDown(self):
        return

    def test_logistic_regresssion(self):
        # train the logistic regression model
        logistic_regression = LogisticRegressionBGD(learning_rate=0.05, num_epochs=1000)
        logistic_regression.fit(self.x, self.y)

        # plot learning curve
        curve = {
            'cost_length': len(logistic_regression.cost),
            'cost': np.log10(logistic_regression.cost),
            'marker': 'o',
            'x_label': 'Epochs',
            'y_label': 'Number of updates',
            'title': 'Logistic regression - Learning rate 0.05'
        }
        Plotter.plot_learning_curve(curve, FilesystemUtils.get_test_resources_plot_file_name(
            'logistic_regression/LogisticRegressionBGD-Learning-Curve.png'))

        # plot decision boundary
        diagram_options = {
            'x_label': 'sepal length [cm]',
            'y_label': 'petal length [cm]',
            'legend': 'upper left'
        }
        Plotter.plot_decision_boundary(self.x, self.y, classifier=logistic_regression, diagram_options=diagram_options,
                                       image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                                           'logistic_regression/LogisticRegressionBGD-Decision-Boundary.png'))


if __name__ == '__main__':
    unittest.main()
