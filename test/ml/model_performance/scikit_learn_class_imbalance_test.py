import numpy as np
from sklearn.utils import resample

from ml.common.data.wdbc_data_reader import WDBCDataReader
from test.ml.common.scikit_learn_test import ScikitLearnTest


class ScikitLearnClassImbalanceTest(ScikitLearnTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_scikit_learn_class_imbalance': cls.load_wdbc_data_set
        }

    def test_scikit_learn_class_imbalance(self):
        print('\n')
        print('Number of class 0 samples before balancing dataset: {}'.format(
            self.x_imbalanced[self.y_imbalanced == 0].shape[0]))
        print('Number of class 1 samples before balancing dataset: {}'.format(
            self.x_imbalanced[self.y_imbalanced == 1].shape[0]))

        # If we were to compute the accuracy of a model that always predicts the majority class (benign, class 0), we
        # would achieve a prediction accuracy of approximately 90 percent
        print('\n')
        print('Just predicting class 0 (benign tumor) on imbalanced dataset scores a {:.4f}% accuracy'.format(
            self.compute_accuracy(self.y_imbalanced)))

        # upsample class 1 sample data so to get the same number of samples as in the class 0 sample data case
        x_upsampled, y_upsampled = resample(
            self.x_imbalanced[self.y_imbalanced == 1], self.y_imbalanced[self.y_imbalanced == 1],
            replace=True, n_samples=self.x_imbalanced[self.y_imbalanced == 0].shape[0], random_state=42
        )
        print('\n')
        print('Number of class 1 samples after upsampling dataset: {}'.format(
            x_upsampled[y_upsampled == 1].shape[0]))

        # now create a balanced dataset
        self.x_balanced = np.vstack((self.x_imbalanced[self.y_imbalanced == 0], x_upsampled))
        self.y_balanced = np.hstack((self.y_imbalanced[self.y_imbalanced == 0], y_upsampled))
        print('\n')
        print('Number of class 0 samples after balancing dataset: {}'.format(
            self.x_balanced[self.y_balanced == 0].shape[0]))
        print('Number of class 1 samples after balancing dataset: {}'.format(
            self.x_balanced[self.y_balanced == 1].shape[0]))

        # now that we have a banced dataset, recompute accuracy and check the difference
        print('\n')
        print('Just predicting class 0 (benign tumor) on balanced dataset scores a {:.4f}% accuracy'.format(
            self.compute_accuracy(self.y_balanced)))

    def compute_accuracy(self, y):
        # just predict class 0
        y_pred = np.zeros(y.shape[0])
        return np.mean(y_pred == y) * 100

    def load_wdbc_data_set(self):
        data_reader = WDBCDataReader()
        wdbc_data_frame = data_reader.get_data()

        x = wdbc_data_frame.loc[:, 2:].values
        y = wdbc_data_frame.loc[:, 1].values

        # create an imbalanced dataset from our breast cancer dataset, which originally consisted of
        # 357 benign tumors (class 0) and 212 malignant tumors (class 1):
        # take all 357 benign tumor samples and stack them with the first 40 malignant samples to create a stark class
        # imbalance

        self.x_imbalanced = np.vstack((x[y == 0], x[y == 1][:40]))
        self.y_imbalanced = np.hstack((y[y == 0], y[y == 1][:40]))
