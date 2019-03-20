import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common.data.wdbc_data_reader import WDBCDataReader
from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnROCAUCTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_roc_auc_on_wdbc_pipeline': cls.load_wdbc_data_set
        }

    def test_scikit_learn_roc_auc_on_wdbc_pipeline(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            LogisticRegression(solver='lbfgs', penalty='l2', C=100.0, random_state=42)
        )

        # create a stratified n_splits-fold cross-validation
        # In this example, we use 3 folds on the training test, actually
        cv = StratifiedKFold(n_splits=3, random_state=42).split(self.x_train, self.y_train)

        false_positive_rates = []
        true_positive_rates = []
        roc_auc_values = []

        # let's use just 10 features to have a more interesting diagram
        x_train_reduced = self.x_train[:, [4, 14]]

        # iterate over the folds to draw the related ROC curve
        for i, (train, test) in enumerate(cv):
            # compute the probabilities predicted by the classifier using the current fold
            probs = wdbc_pipeline.fit(x_train_reduced[train], self.y_train[train]).predict_proba(x_train_reduced[test])

            # computer ROC curve arrays
            fpr, tpr, thresholds = roc_curve(self.y_train[test], probs[:, 1], pos_label=1)

            # compute AUC (Area Under Curve)
            roc_auc = auc(fpr, tpr)

            false_positive_rates.append(fpr)
            true_positive_rates.append(tpr)
            roc_auc_values.append(roc_auc)

        # display results
        image_file_path = FilesystemUtils.get_test_resources_plot_file_name(
            'model_performance/ModelPerformance-ScikitLearn-ROC_AUC.png'
        )
        Plotter.plot_roc_auc(
            np.asarray(false_positive_rates), np.asarray(true_positive_rates), np.asarray(roc_auc_values),
            image_file_path=image_file_path
        )

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
