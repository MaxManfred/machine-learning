import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnKNestedCVTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_nested_cv_by_svm': cls.load_wdbc_data_set,
            'test_scikit_learn_nested_cv_by_decision_tree': cls.load_wdbc_data_set
        }

    def test_scikit_learn_nested_cv_by_svm(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            SVC(random_state=42)
        )

        # In nested cross-validation, we have an outer k-fold cross-validation loop (with 5 folds in this example) to
        # split the data into training and test folds, and an inner loop (with 2 folds in this example) that is used
        # to select the model using k-fold cross-validation on the training fold.
        # After model selection, the test fold is then used to evaluate the model performance.
        # This is an example of a 5x2 nested cross-validation

        num_inner_folds = 2
        num_outer_folds = 5

        # prepare a grid search with num_inner_folds-fold cross-validation inner loop to select the best model
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [
            {
                'svc__C': param_range,
                'svc__kernel': ['linear']
            },
            {
                'svc__C': param_range,
                'svc__gamma': param_range,
                'svc__kernel': ['rbf']
            }
        ]
        gs = GridSearchCV(
            estimator=wdbc_pipeline,
            param_grid=param_grid,
            scoring='accuracy',
            cv=num_inner_folds,
            iid=False, n_jobs=-1
        )
        # use a num_outer_folds-fold cross-validation outer loop to evaluate the model the performance of model selected
        # at previous step on the test fold
        scores = cross_val_score(gs, self.x_train, self.y_train, scoring='accuracy', cv=num_outer_folds)

        # display the result
        print(' ')
        print('CV accuracy for SVM pipeline: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))
        print(' ')

    def test_scikit_learn_nested_cv_by_decision_tree(self):
        # create a decision tree to test
        decision_tree = DecisionTreeClassifier(random_state=42)

        num_inner_folds = 2
        num_outer_folds = 5

        # prepare a grid search with num_inner_folds-fold cross-validation inner loop to select the best model
        param_range = [1, 2, 3, 4, 5, 6, 7, None]
        param_grid = [
            {
                'max_depth': param_range
            }
        ]
        gs = GridSearchCV(
            estimator=decision_tree,
            param_grid=param_grid,
            scoring='accuracy',
            cv=num_inner_folds,
            n_jobs=-1
        )
        # use a num_outer_folds-fold cross-validation outer loop to evaluate the model the performance of model selected
        # at previous step on the test fold
        scores = cross_val_score(gs, self.x_train, self.y_train, scoring='accuracy', cv=num_outer_folds)

        # display the result
        print(' ')
        print('CV accuracy for decision tree: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))
        print(' ')

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
