import numpy as np
from numpy.random import RandomState

from ml.common.classification.classifier import Classifier


class Perceptron(Classifier):
    """
    Perceptron classifier.

    :ivar learning_rate:
        learning rate (between 0.0 and 1.0)
    :ivar num_epochs:
        passes over the training dataset
    :ivar weight_init_seed:
        random number generator seed for random weight initialization
    :ivar weights:
        1d-array of trained weights
    :ivar cost:
        list of number of misclassifications in each epoch
    """
    weights: np.ndarray

    def __init__(self, learning_rate: float = 0.01, num_epochs: int = 50, weight_init_seed: int = 1) -> None:
        """
        Constructor

        :rtype: None
        :type learning_rate: float
        :param learning_rate: learning rate (between 0.0 and 1.0)
        :type num_epochs: int
        :param num_epochs: passes over the training dataset
        :type weight_init_seed: int
        :param weight_init_seed: random number generator seed for random weight initialization
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_seed = weight_init_seed
        self.cost = []
        self.weights = None

    def activation(self, x):
        """Compute linear activation"""
        return x

    def fit(self, x: np.matrix, y: np.matrix) -> object:
        """
        Trains the perceptron

        :type x: np.matrix[num_samples, num_features]
        :param x: training set, where num_samples is the number of samples and num_features is the number of features
        :type y: np.matrix[num_samples]
        :param y: training set labels, where num_samples is the number of samples
        :rtype: Perceptron
        :return: self
        """
        random_number_generator: RandomState = RandomState(self.weight_init_seed)
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])

        for i in range(self.num_epochs):
            print('Training epoch ' + str(i + 1))

            errors: int = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))

                self.weights[1:] += update * xi
                self.weights[0] += update

                errors += int(update != 0.0)
            self.cost.append(errors)

        return self

    def predict(self, x: np.matrix) -> np.ndarray:
        """
        Predicts class labels by applying the unit step activation function

        :type x: np.matrix[num_test_elements, num_features]
        :param x: test set, where num_test_elements is the number of elements to test and num_features is the number of
                  features
        :rtype: np.core.multiarray
        :return: the predicted class labels
        """
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)
