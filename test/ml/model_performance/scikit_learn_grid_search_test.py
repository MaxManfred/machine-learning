from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnGridSearchTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_grid_search_on_wdbc_pipeline': cls.load_wdbc_data_set
        }

    def test_scikit_learn_grid_search_on_wdbc_pipeline(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            SVC(random_state=42)
        )

        # The approach of grid search is quite simple: it is a brute-force exhaustive search paradigm where we specify
        # a list of values for different hyperparameters, and the computer evaluates the model performance for each
        # combination of those to obtain the optimal combination of values from this set:

        num_folds = 10
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
            cv=num_folds,
            iid=False, n_jobs=-1
        )
        gs = gs.fit(X=self.x_train, y=self.y_train)

        # display results
        print('Best {}-fold cv accuracy score: {}'.format(num_folds, gs.best_score_))
        print('Best {}-fold cv hyper-parameters: {}'.format(num_folds, gs.best_params_))

        # Using the preceding code, we initialized a GridSearchCV object from the sklearn.model_selection module to
        # train and tune a Support Vector Machine (SVM) pipeline.
        # We set the param_grid parameter of GridSearchCV to a list of dictionaries to specify the parameters that
        # we want to tune.
        # For the linear SVM, we only evaluated the inverse regularization parameter C; for the RBF kernel SVM,
        # we tuned both the svc__C and svc__gamma parameter. Note that the svc__gamma parameter is specific to
        # kernel SVMs.

        # Finally, we will use the independent test dataset to estimate the performance of the best-selected model,
        # which is available via the best_estimator_ attribute of the GridSearchCV object
        best_classifier = gs.best_estimator_
        best_classifier.fit(X=self.x_test, y=self.y_test)
        print('Test accuracy score of best model: {:.3f}'.format(best_classifier.score(X=self.x_test, y=self.y_test)))

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
