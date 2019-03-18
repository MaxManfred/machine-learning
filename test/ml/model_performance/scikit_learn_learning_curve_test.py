import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common.data.wdbc_data_reader import WDBCDataReader
from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnLeaningCurvesTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_learning_curve_on_wdbc_pipeline': cls.load_wdbc_data_set
        }

    def test_scikit_learn_learning_curve_on_wdbc_pipeline(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            LogisticRegression(solver='lbfgs', random_state=42)
        )

        # prepare learning curve

        # Via the train_sizes parameter in the learning_curve function, we can control the absolute or relative number
        # of training samples that are used to generate the learning curves.
        # Here, we set train_sizes=np.linspace(0.1, 1.0, 10) to use 10 evenly spaced, relative intervals for the
        # training set sizes. By default, the learning_curve function uses stratified k-fold cross-validation to
        # calculate the cross-validation accuracy of a classifier, and we set k=num_folds via the cv parameter for
        # k-fold stratified cross-validation.
        # Finally, we plot the diagram using a helper function.

        num_folds = 10
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=wdbc_pipeline,
            X=self.x_train, y=self.y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=num_folds,
            n_jobs=-1
        )

        image_file_path = FilesystemUtils.get_test_resources_plot_file_name(
            'model_performance/ModelPerformance-ScikitLearn-LearningCurves.png'
        )
        Plotter.plot_performance_curves(train_sizes, train_scores, test_scores, image_file_path=image_file_path)

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
