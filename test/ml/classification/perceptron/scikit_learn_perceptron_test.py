import unittest

import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier

from ml.common.plot.plotter import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnPerceptronTest(ScikitLearnTest):

    def test_scikit_learn_perceptron(self):
        # Train the perceptron.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        perceptron = Perceptron(n_iter=40, eta0=0.1, random_state=1)
        perceptron.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(perceptron,
                                  image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                                      'perceptron/Perceptron-ScikitLearn-Decision-Boundary.png'))

    def test_scikit_learn_perceptron_by_SGDClassifier(self):
        # Train the SVM.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        # Sometimes our datasets are too large to fit into computer memory, thus, scikit-learn also offers alternative
        # implementations viaThe SGDClassifier class, which also supports online learning via the partial_fit method.
        # The concept behind the SGDClassifier class is similar to the stochastic gradient algorithm
        perceptron = SGDClassifier(loss='perceptron')
        perceptron.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(perceptron,
                                  FilesystemUtils.get_test_resources_plot_file_name(
                                      'perceptron/Perceptron-ScikitLearn-Classifier-Decision-Boundary.png'))

    def predict_and_evaluate(self, perceptron: Perceptron, image_file_path: str = None):
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
                                       image_file_path=image_file_path)


if __name__ == '__main__':
    unittest.main()
