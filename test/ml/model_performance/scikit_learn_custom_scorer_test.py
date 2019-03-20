from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnCustomScorerTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_custom_scorer_by_svm': cls.load_wdbc_data_set
        }

    def test_scikit_learn_custom_scorer_by_svm(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            SVC(random_state=42)
        )

        # create a custom scorer
        # Remember that the positive class in scikit-learn is the class that is labeled as class 1.
        # If we want to specify a different positive label, we can construct our own scorer via the make_scorer
        # function, which we can then directly provide as an argument to the scoring parameter in GridSearchCV
        # (in this example, using the f1_score as a metric)
        scorer = make_scorer(f1_score, pos_label=0)

        # We can use a different scoring metric than accuracy in the GridSearchCV via the scoring parameter.
        # A complete list of the different values that are accepted by the scoring parameter can be found at
        # http://scikit-learn.org/stable/modules/model_evaluation.html.

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
            scoring=scorer,
            cv=num_folds,
            iid=False, n_jobs=-1
        )
        gs = gs.fit(self.x_train, self.y_train)

        # display results
        print('Best {}-fold cv accuracy score: {}'.format(num_folds, gs.best_score_))
        print('Best {}-fold cv hyper-parameters: {}'.format(num_folds, gs.best_params_))

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
