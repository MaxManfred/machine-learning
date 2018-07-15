from abc import abstractmethod
import numpy as np


class Classifier(object):

    @abstractmethod
    def train(self, X: np.matrix, Y: np.matrix):
        pass

    @abstractmethod
    def predict(self, X: np.matrix) -> np.ndarray:
        pass
