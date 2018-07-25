from abc import abstractmethod
import numpy as np


class Classifier(object):

    @abstractmethod
    def train(self, X: np.matrix, Y: np.matrix):
        pass

    @abstractmethod
    def predict(self, X: np.matrix) -> np.ndarray:
        pass

    @abstractmethod
    def activation(self, X):
        pass

    def net_input(self, X: np.matrix) -> np.matrix:
        """
        Calculate net input

        :return:
        :type X: np.matrix[num_samples, num_features]
        :param X: training set, where num_samples is the number of samples and num_features is the number of features
        :rtype: np.matrix[num_samples, num_features]
        :return: the linear combination of the weights with the training set samples
        """
        return np.dot(X, self.weights[1:]) + self.weights[0]
