import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnMacroMicroAveragingTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_macro_averaging_by_logistic_regression': cls.load_wdbc_data_set,
            'test_scikit_learn_micro_averaging_by_logistic_regression': cls.load_wdbc_data_set
        }

    def test_scikit_learn_macro_averaging_by_logistic_regression(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            SVC(random_state=42)
        )

        # The scoring metrics like TP, TN, FP, FN, precision, recall, f1-score are specific to binary classification
        # systems. However, scikit-learn also implements macro and micro averaging methods to extend those scoring
        # metrics to multiclass problems via One-versus-All (OvA) classification. The micro-average is calculated
        # from the individual TPs, TNs, FPs, and FNs of the system. For example
        # pre_micro = (TP1 + ... + TPk) / [ (TP1 + ... + TPk) + (FP1 + ... + FPk) ]
        # The macro-average is simply calculated as the average scores of the different systems. For example
        # pre_macro = (PRE1 + ... + PREk) / k
        # Micro-averaging is useful if we want to weight each instance or prediction equally, whereas macro-averaging
        # weights all classes equally to evaluate the overall performance of a classifier with regard to the most
        # frequent class labels.
        #
        # If we are using binary performance metrics to evaluate multiclass classification models in scikit-learn,
        # a normalized or weighted variant of the macro-average is used by default. The weighted macro-average is
        # calculated by weighting the score of each class label by the number of true instances when calculating
        # the average. The weighted macro-average is useful if we are dealing with class imbalances, that is,
        # different numbers of instances for each label.
        # While the weighted macro-average is the default for multiclass problems in scikit-learn, we can specify the
        # averaging method via the average parameter inside the different scoring functions that we import from the
        # sklearn.metrics module, for example, the precision_score or make_scorer functions

        # create a custom scorer
        pre_macro_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='macro')

        num_folds = 10
        scores = cross_val_score(
            estimator=wdbc_pipeline, scoring=pre_macro_scorer, X=self.x_train, y=self.y_train, cv=num_folds, n_jobs=-1
        )

        # display results
        print(' ')
        print('{}-fold CV macro precision scores: {}'.format(num_folds, scores))
        print('{}-fold CV macro precision: {:.3f} +/- {:.3f}'.format(num_folds, np.mean(scores), np.std(scores)))
        print(' ')

    def test_scikit_learn_micro_averaging_by_logistic_regression(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            LogisticRegression(solver='lbfgs', random_state=42)
        )

        # create a custom scorer
        pre_micro_scorer = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average='micro')

        num_folds = 10
        scores = cross_val_score(
            estimator=wdbc_pipeline, scoring=pre_micro_scorer, X=self.x_train, y=self.y_train, cv=num_folds, n_jobs=-1
        )

        # display results
        print(' ')
        print('{}-fold CV micro precision scores: {}'.format(num_folds, scores))
        print('{}-fold CV micro precision: {:.3f} +/- {:.3f}'.format(num_folds, np.mean(scores), np.std(scores)))
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
