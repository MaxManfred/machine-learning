import unittest

import numpy as np
from pydotplus import graph_from_dot_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnDecisionTreeTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_draw_impurity_criteria_types': cls.dummy_load_data_set
        }

    def test_draw_impurity_criteria_types(self):
        # For a visual comparison of the three different impurity criteria (Entropy, Gini, Misclassification Error),
        # let us plot the impurity indices for the probability range [0, 1] for class 1.
        # Note that we will also add a scaled version of the entropy (entropy / 2) to observe that the Gini impurity is
        # an intermediate measure between entropy and the classification error.

        x_range = np.arange(start=0.0, stop=1.0, step=0.01, dtype=float)

        # compute entropy and scaled entropy
        ent = [self.entropy(p) if p != 0 else None for p in x_range]
        sc_ent = [e * 0.5 if e else None for e in ent]

        # compute gini
        gn = [self.gini(p) for p in x_range]

        # computer misclassification error
        err = [self.misclassification_error(p) for p in x_range]

        Plotter.plot_impurity_criteria(x_range, entropy=ent, scaled_entropy=sc_ent, gini=gn,
                                       misclassification_error=err,
                                       image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                                           'decision_tree/DecisionTree-ScikitLearn-Impurity-Criteria.png'))

    def test_scikit_learn_decision_tree(self):
        # Train the decision tree.
        # Most algorithms in scikit-learn already support multiclass classification via the One-versus-Rest (OvR) method
        tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
        tree.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(tree,
                                  image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                                      'decision_tree/DecisionTree-ScikitLearn-Decision-Boundary.png'))

        # save the tree as a digram
        dot_data = export_graphviz(tree, filled=True, rounded=True, class_names=['Setosa', 'Versicolor', 'Virginica'],
                                   feature_names=['petal length', 'petal width'], out_file=None)
        graph = graph_from_dot_data(dot_data)
        graph.write_png(FilesystemUtils.get_test_resources_plot_file_name(
            'decision_tree/DecisionTree-ScikitLearn-Tree-Diagram.png'))

    def test_scikit_learn_random_forest(self):
        forest = RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=2)
        forest.fit(self.x_train, self.y_train)

        self.predict_and_evaluate(forest,
                                  image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                                      'decision_tree/RandomForest-ScikitLearn-Decision-Boundary.png'))

    def predict_and_evaluate(self, decision_tree: DecisionTreeClassifier, image_file_path: str = None):
        # Run predictions and count the number of misclassified examples
        y_pred = decision_tree.predict(self.x_test)
        print('Misclassified samples: %d' % (self.y_test != y_pred).sum())
        # Evaluate model accuracy
        # Each classifier in scikit-learn has a score method, which computes a classifier's prediction accuracy by
        # combining the predict call with the accuracy_score call
        print('Accuracy: %.2f' % decision_tree.score(self.x_test, self.y_test))

        # Show decision boundary
        diagram_options = {
            'x_label': 'petal length [standardized]',
            'y_label': 'petal width [standardized]',
            'legend': 'upper left',
            'draw_test_samples': range(105, 150)
        }
        x_combined_std = np.vstack((self.x_train, self.x_test))
        y_combined = np.hstack((self.y_train, self.y_test))

        Plotter.plot_decision_boundary(x_combined_std, y_combined, decision_tree, diagram_options,
                                       image_file_path=image_file_path)

    @staticmethod
    def entropy(p: float):
        return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

    @staticmethod
    def gini(p: float):
        return p * (1 - p) + (1 - p) * (1 - (1 - p))

    @staticmethod
    def misclassification_error(p: float):
        return 1 - np.max([p, 1 - p])

    def dummy_load_data_set(self):
        pass


if __name__ == '__main__':
    unittest.main()
