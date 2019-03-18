from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnPipelineTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_wdbc_pipeline': cls.load_wdbc_data_set
        }

    def test_scikit_learn_wdbc_pipeline(self):
        # Learning algorithms require input features on the same scale for optimal performance.
        # Thus, we need to standardize the columns in the Breast Cancer Wisconsin dataset before we can feed them to a
        # linear classifier, such as logistic regression.
        # Furthermore, we assume that we want to compress our data from the initial 30 dimensions onto a lower
        # two-dimensional subspace via Principal Component Analysis (PCA),
        # a feature extraction technique for dimensionality reduction

        # Instead of going through the fitting and transformation steps for the training and test datasets separately,
        # we can chain the StandardScaler, PCA, and LogisticRegression objects in a pipeline

        wdbc_pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=2),
            LogisticRegression(solver='lbfgs', random_state=42)
        )

        # train
        wdbc_pipeline.fit(self.x_train, self.y_train)

        # predict
        # y_pred = wdbc_pipeline.predict(self.x_test)

        print('Test Accuracy: %.3f' % wdbc_pipeline.score(self.x_test, self.y_test))

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # Before we construct our first model pipeline, let us divide the dataset into a separate training dataset
        # (80 percent of the data) and a separate test dataset (20 percent of the data)

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
