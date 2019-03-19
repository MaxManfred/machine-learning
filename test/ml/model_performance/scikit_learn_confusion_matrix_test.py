from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.common.data.wdbc_data_reader import WDBCDataReader
from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnConfusionMatrixTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_confusion_matrix_by_svm': cls.load_wdbc_data_set
        }

    def test_scikit_learn_confusion_matrix_by_svm(self):
        # create a pipeline to test
        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            SVC(random_state=42)
        )

        wdbc_pipeline.fit(self.x_train, self.y_train)

        y_pred = wdbc_pipeline.predict(self.x_test)

        cm = confusion_matrix(y_true=self.y_test, y_pred=y_pred)

        # display confusion matrix
        print('Confusion matrix')
        print(cm)

        # display metrics
        print('Precision: {:.3f}'.format(precision_score(y_true=self.y_test, y_pred=y_pred)))
        print('Recall: {:.3f}'.format(recall_score(y_true=self.y_test, y_pred=y_pred)))
        print('F1: {:.3f}'.format(f1_score(y_true=self.y_test, y_pred=y_pred)))

        # The array that was returned after executing the code provides us with information about the different types of
        # error the classifier made on the test dataset.

        image_file_path = FilesystemUtils.get_test_resources_plot_file_name(
            'model_performance/ModelPerformance-ScikitLearn-ConfusionMatrix.png'
        )
        Plotter.plot_confusion_matrix(cm, image_file_path=image_file_path)

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
