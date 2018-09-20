import pandas as pd
import numpy as np

from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.scikit_learn_test import ScikitLearnTest


class WineDatasetTest(ScikitLearnTest):

    def load_wine_data_set(self):
        # Load a non linearly separable dataset
        # There are 13 different features in the Wine dataset, describing the chemical properties of the 178 wine
        # samples, and each sample belongs to one of three different classes, 1, 2, and 3, which refer to the three
        # different types of grape grown in the same region in Italy but derived from different wine cultivars, as
        # described in the dataset summary (https://archive. ics.uci.edu/ml/machine-learning-databases/wine/wine.names).

        # load wine data set
        df = pd.read_csv(FilesystemUtils.get_resources_data_file_name('wine/wine.data'), header=None)

        # create headers
        df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                      'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                      'OD280/OD315 of diluted wines', 'Proline']

        # print sample info:
        # values of class labels
        print('Class labels', np.unique(df['Class label']))
        # 10% of the shuffled data set samples with respective class labels (shuffling is necessary as rows are sorted
        # according to class label values in increasing order
        print(df.sample(frac=0.1).to_string())

        # split training and test set
        # separate class labels from features
        self.x, self.y = df.iloc[:, 1:].values, df.iloc[:, 0].values

        # save column names for later usage
        self.df_columns = df.columns
