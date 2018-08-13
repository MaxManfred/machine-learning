import unittest

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from ml.common.plot.plotter import Plotter
from tests.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnSVMTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_svm_nonlinear': cls.load_svm_nonlinear_data_set
        }

    def test_scikit_learn_svm(self):
        # Train the SVM.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        svm = SVC(kernel='linear', C=1.0, random_state=1)
        svm.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(svm, '../../resources/images/SVM-ScikitLearn-Decision-Boundary.png')

    def test_scikit_learn_svm_by_SGDClassifier(self):
        # Train the SVM.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        # Sometimes our datasets are too large to fit into computer memory, thus, scikit-learn also offers alternative
        # implementations viaThe SGDClassifier class, which also supports online learning via the partial_fit method.
        # The concept behind the SGDClassifier class is similar to the stochastic gradient algorithm
        svm = SGDClassifier(loss='hinge')
        svm.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(svm, '../../resources/images/SVM-ScikitLearn-Classifier-Decision-Boundary.png')

    def test_scikit_learn_svm_nonlinear(self):
        svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
        svm.fit(self.x_train, self.y_train)

        diagram_options = {
            'x_label': 'feature 1',
            'y_label': 'feature 2',
            'legend': 'best'
        }
        Plotter.plot_decision_boundary(self.x_train, self.y_train, svm, diagram_options,
                                       image_file_path='../../resources/images/SVM-ScikitLearn-NonLinear-Decision-Boundary.png')

    def load_svm_nonlinear_data_set(self):
        # Load a non linearly separable dataset
        np.random.seed(1)
        x_xor = np.random.randn(200, 2)
        y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
        y_xor = np.where(y_xor, 1, -1)

        # plotter data and save it to file
        Plotter.plot_svm_nonlinear_data_set(x_xor, y_xor,
                                            '../../resources/images/SVM-ScikitLearn-NonLinear-Training-Set.png')

        self.x_train = x_xor
        self.y_train = y_xor

    def predict_and_evaluate(self, svm, image_file_path: str = None):
        # Run predictions and count the number of misclassified examples
        y_pred = svm.predict(self.x_test)
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())
        # Evaluate model accuracy
        # Each classifier in scikit-learn has a score method, which computes a classifier's prediction accuracy by
        # combining the predict call with the accuracy_score call
        print('Accuracy: %.2f' % svm.score(self.x_test, self.y_test))
        # Show decision boundary
        diagram_options = {
            'x_label': 'petal length [standardized]',
            'y_label': 'petal width [standardized]',
            'legend': 'upper left',
            'draw_test_samples': range(105, 150)
        }
        x_combined_std = np.vstack((self.x_train, self.x_test))
        y_combined = np.hstack((self.y_train, self.y_test))
        Plotter.plot_decision_boundary(x_combined_std, y_combined, svm, diagram_options,
                                       image_file_path=image_file_path)


if __name__ == '__main__':
    unittest.main()
