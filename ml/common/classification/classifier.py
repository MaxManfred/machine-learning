from abc import abstractmethod

import numpy as np


class Classifier(object):

    @abstractmethod
    def fit(self, x: np.matrix, y: np.matrix):
        pass

    @abstractmethod
    def predict(self, x: np.matrix) -> np.ndarray:
        pass

    @abstractmethod
    def activation(self, x):
        pass

    def net_input(self, x: np.matrix) -> np.matrix:
        """
        Calculate net input

        :return:
        :type x: np.matrix[num_samples, num_features]
        :param x: training set, where num_samples is the number of samples and num_features is the number of features
        :rtype: np.matrix[num_samples, num_features]
        :return: the linear combination of the weights with the training set samples
        """
        return np.dot(x, self.weights[1:]) + self.weights[0]
