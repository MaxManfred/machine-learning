import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from definitions import RESOURCES_DATA_DIR
from ml.common.data.data_reader import DataReader

"""
    The Breast Cancer Wisconsin dataset (WDBC in short) contains 569 samples of malignant and benign tumor cells.
    The first two columns in the dataset store the unique ID numbers of the samples and the corresponding diagnoses
    (M = malignant, B = benign), respectively.
    Columns 3-32 contain 30 real-valued features that have been computed from digitized images of the cell nuclei,
    which can be used to build a model to predict whether a tumor is benign or malignant.
    The Breast Cancer Wisconsin dataset has been deposited in the UCI Machine Learning Repository, and more detailed
    information about this dataset can be found at
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic).
"""


class WDBCDataReader(DataReader):
    DATASET_RELATIVE_FILE_PATH: str = 'wdbc/wdbc.csv'

    DATASET_NAMES_RELATIVE_FILE_PATH: str = 'wdbc/wdbc.names'

    DATASET_REMOTE_URL: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

    DATASET_NAMES_REMOTE_URL: str = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names'

    def get_data(self):
        # check data file is existing
        dataset_path = os.path.join(RESOURCES_DATA_DIR, self.DATASET_RELATIVE_FILE_PATH)
        dataset_names_path = os.path.join(RESOURCES_DATA_DIR, self.DATASET_NAMES_RELATIVE_FILE_PATH)

        if not os.path.isfile(dataset_path):
            # download dataset from remote
            print('Downloading dataset...')
            import urllib.request
            urllib.request.urlretrieve(self.DATASET_REMOTE_URL, dataset_path, self.download_progress_monitor)
            print('\nDone!')
        else:
            print('Dataset has already been downloaded')

        if not os.path.isfile(dataset_names_path):
            # download dataset names from remote
            print('Downloading dataset names...')
            import urllib.request
            urllib.request.urlretrieve(self.DATASET_NAMES_REMOTE_URL, dataset_names_path,
                                       self.download_progress_monitor)
            print('\nDone!')
        else:
            print('Dataset names has already been downloaded')

        # load dataset as a data frame
        wdbc_data_frame = self._load_data(relative_file_path=self.DATASET_RELATIVE_FILE_PATH, use_header=False)

        # use a LabelEncoder object to transform the class labels from their original string representation
        # ('M' and 'B') into integers
        le = LabelEncoder()
        y = wdbc_data_frame.iloc[:, 1].values
        wdbc_data_frame.iloc[:, 1] = pd.Series(le.fit_transform(y))

        # After encoding the class labels (diagnosis) in an array y, the malignant tumors are now represented as class 1
        # and the benign tumors are represented as class 0, respectively.

        print(' ')
        print('Found classes {} on column at index 1'.format(le.classes_))

        print(' ')
        print(wdbc_data_frame.head(10))

        return wdbc_data_frame


if __name__ == "__main__":
    WDBCDataReader().get_data()
