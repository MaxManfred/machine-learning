import pandas as pd
from pandas import DataFrame
import numpy as np


class IrisDataReader(object):

    def __init__(self, data_file_path: str = None, num_samples: int = 150) -> None:
        """
        Constructor

        :type num_samples: int
        :param num_samples: the number of Iris data rows to load
        :rtype: None
        :return None
        """
        # Read Iris dataset
        # get last lines
        self.iris_data_frame: DataFrame = pd.read_csv(data_file_path, header=None).tail(num_samples)

    def get_data(self) -> (np.matrix, np.matrix):

        # extract sepal length and petal length
        X = self.iris_data_frame.iloc[0:100, [0, 2]].values

        # select setosa and versicolor
        Y = self.iris_data_frame.iloc[0:100, 4].values
        Y = np.where(Y == 'Iris-setosa', -1, 1)

        return X, Y

