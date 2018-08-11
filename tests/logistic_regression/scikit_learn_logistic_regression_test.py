import unittest

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.common.plot.plotter import Plotter


class ScikitLearnLogisticRegressionTest(unittest.TestCase):

    def setUp(self):
        # Loading the Iris dataset from scikit-learn.
        # The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.
        iris = datasets.load_iris()
        x = iris.data[:, [2, 3]]
        y = iris.target
        print('Class labels:', np.unique(y))

        # plotter data and save it to file
        Plotter.plot_data_set(x, '../../resources/images/Perceptron-ScikitLearn-Training-Set.png')

        # Splitting data into 70% training and 30% test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
        print('Labels counts in y:', np.bincount(y))
        print('Labels counts in y_train:', np.bincount(y_train))
        print('Labels counts in y_test:', np.bincount(y_test))

        # Standardize features
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        sc.fit(x_test)
        x_test_std = sc.transform(x_test)

        self.x_train = x_train_std
        self.y_train = y_train

        self.x_test = x_test_std
        self.y_test = y_test

    def tearDown(self):
        return

    def test_scikit_learn_logistic_regression(self):
        # Train the logistic regression.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        # Parameter C control regularization and C = 1 / lambda, where the regularization term is (lambda / 2) * exp(norm(weights), 2)
        logistic_regression = LogisticRegression(C=100.0, random_state=1)
        logistic_regression.fit(self.x_train, self.y_train)

        # Run predictions and count the number of misclassified examples
        y_pred = logistic_regression.predict(self.x_test)
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())

        # Compute the probabilities for the first 3 samples to belong to each of the classes
        probs = logistic_regression.predict_proba(self.x_test[:3, :])
        print('Computed probabilities for the first 3 samples:\n', probs)

        # Evaluate model accuracy
        # Each classifier in scikit-learn has a score method, which computes a classifier's prediction accuracy by
        # combining the predict call with the accuracy_score call
        print('Accuracy: %.2f' % logistic_regression.score(self.x_test, self.y_test))

        # Show decision boundary
        diagram_options = {
            'x_label': 'petal length [standardized]',
            'y_label': 'petal width [standardized]',
            'legend': 'upper left',
            'draw_test_samples': range(105, 150)
        }
        x_combined_std = np.vstack((self.x_train, self.x_test))
        y_combined = np.hstack((self.y_train, self.y_test))
        Plotter.plot_decision_boundary(x_combined_std, y_combined, logistic_regression, diagram_options,
                                       image_file_path='../../resources/images/LogisticRegression-ScikitLearn-Decision-Boundary.png')

    def test_scikit_learn_logistic_regression_regularization(self):
        weights, params = [], []
        for c in np.arange(-5, 5):
            logistic_regression = LogisticRegression(C=10. ** c, random_state=1)
            logistic_regression.fit(self.x_train, self.y_train)
            weights.append(logistic_regression.coef_[1])
            params.append(10. ** c)
        weights = np.array(weights)

        curve = {
            'x_label': 'inverse-regularization parameter C',
            'y_label': 'weight coefficient for class 1',
            'title': 'regularizarion curves',
            'legend': 'upper left'
        }

        Plotter.draw_regularizatiion_curves(curve, params, weights,
                                            image_file_path='../../resources/images/LogisticRegression-ScikitLearn-Regularization-Curves.png')


if __name__ == '__main__':
    unittest.main()
