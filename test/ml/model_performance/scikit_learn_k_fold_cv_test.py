import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnKFoldCVTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_k_fold_cv_on_wdbc_pipeline': cls.load_wdbc_data_set
        }

    def test_scikit_learn_k_fold_cv_on_wdbc_pipeline(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            LogisticRegression(solver='lbfgs', random_state=42)
        )

        # Scikit-learn also implements a k-fold cross-validation scorer, which allows us to evaluate our model using
        # stratified k-fold
        # if cross - validation:
        # Stratification ensures that both training and test datasets have the same class proportions as the original
        # dataset

        # An extremely useful feature of the cross_val_score approach is that we can distribute the evaluation of the
        # different folds across multiple CPUs on our machine.
        # If we set the n_jobs parameter to 1, only one CPU will be used to evaluate the performances.
        # However, by setting n_jobs=2, we could distribute the 10 rounds of cross-validation to two CPUs
        # (if available on our machine), and by setting n_jobs=-1, we can use all available CPUs on our machine to do
        # the computation in parallel.

        num_folds = 10
        scores = cross_val_score(estimator=wdbc_pipeline, X=self.x_train, y=self.y_train, cv=num_folds, n_jobs=-1)

        print('{}-fold CV accuracy scores: {}'.format(num_folds, scores))
        print('{}-fold CV accuracy: {:.3f} +/- {:.3f}'.format(num_folds, np.mean(scores), np.std(scores)))

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
