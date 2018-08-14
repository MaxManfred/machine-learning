import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ml.common.plot.plotter import Plotter
from tests.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnKNNTest(ScikitLearnTest):

    def test_scikit_learn_knn(self):
        # Train the perceptron.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        knn.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(knn,
                                  image_file_path='../../resources/images/KNN-ScikitLearn-Decision-Boundary.png')

    def predict_and_evaluate(self, knn: KNeighborsClassifier, image_file_path: str = None):
        # Run predictions and count the number of misclassified examples
        y_pred = knn.predict(self.x_test)
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())
        # Evaluate model accuracy
        # Each classifier in scikit-learn has a score method, which computes a classifier's prediction accuracy by
        # combining the predict call with the accuracy_score call
        print('Accuracy: %.2f' % knn.score(self.x_test, self.y_test))
        # Show decision boundary
        diagram_options = {
            'x_label': 'petal length [standardized]',
            'y_label': 'petal width [standardized]',
            'legend': 'upper left',
            'draw_test_samples': range(105, 150)
        }
        x_combined_std = np.vstack((self.x_train, self.x_test))
        y_combined = np.hstack((self.y_train, self.y_test))
        Plotter.plot_decision_boundary(x_combined_std, y_combined, knn, diagram_options,
                                       image_file_path=image_file_path)


if __name__ == '__main__':
    unittest.main()
