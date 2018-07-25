import numpy as np
from numpy.random.mtrand import RandomState

from ml.common.classification import Classifier


class AdalineGD(Classifier):

    def __init__(self, learning_rate: float = 0.01, num_epochs: int = 50, weight_init_seed: int = 1) -> None:
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_seed = weight_init_seed
        self.cost = []
        self.weights = None

    def activation(self, X):
        """Compute linear activation"""
        return X

    def train(self, X: np.matrix, Y: np.matrix):
        random_number_generator: RandomState = RandomState(self.weight_init_seed)
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.num_epochs):
            print('Training epoch ' + str(i + 1))

            # Please note that the "activation" method has no effect in the code since it is simply an identity
            # function.
            # We could write output = self.net_input(X) directly instead.
            # The purpose of the activation is more conceptual, i.e., in the case of logistic regression (as we will see
            # later),
            # we could change it to a sigmoid function to implement a logistic regression classifier.
            output = self.activation(self.net_input(X))
            errors = (Y - output)

            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

            self.cost.append((errors**2).sum() / 2.0)

        return self

    def predict(self, X: np.matrix) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
