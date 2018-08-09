import unittest

import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.common.plot.plotter import Plotter


class PerceptronTest(unittest.TestCase):

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

    def test_scikit_learn_perceptron(self):
        # Train the perceptron.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        perceptron = Perceptron(n_iter=40, eta0=0.1, random_state=1)
        perceptron.fit(self.x_train, self.y_train)

        # Run predictions and count the number of misclassified examples
        y_pred = perceptron.predict(self.x_test)
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())

        # Evaluate model accuracy
        # Each classifier in scikit-learn has a score method, which computes a classifier's prediction accuracy by
        # combining the predict call with the accuracy_score call
        print('Accuracy: %.2f' % perceptron.score(self.x_test, self.y_test))

        # Show decision boundary
        diagram_options = {
            'x_label': 'petal length [standardized]',
            'y_label': 'petal width [standardized]',
            'legend': 'upper left',
            'draw_test_samples': range(105, 150)
        }
        x_combined_std = np.vstack((self.x_train, self.x_test))
        y_combined = np.hstack((self.y_train, self.y_test))
        Plotter.plot_decision_boundary(x_combined_std, y_combined, perceptron, diagram_options,
                                       image_file_path='../../resources/images/Perceptron-ScikitLearn-Decision-Boundary.png')


if __name__ == '__main__':
    unittest.main()
