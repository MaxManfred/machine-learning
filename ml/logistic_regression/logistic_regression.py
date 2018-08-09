import numpy as np
from numpy.random.mtrand import RandomState

from ml.common.classification import Classifier


class LogisticRegressionBGD(Classifier):

    def __init__(self, learning_rate: float = 0.01, num_epochs: int = 50, weight_init_seed: int = 1) -> None:
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_seed = weight_init_seed
        self.cost = []
        self.weights = None

    def fit(self, x: np.matrix, y: np.matrix):
        random_number_generator: RandomState = RandomState(self.weight_init_seed)
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])

        for i in range(self.num_epochs):
            output = self.activation(self.net_input(x))
            errors = (y - output)

            # update weights
            self.weights[1:] += self.learning_rate * x.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

            # update cost
            self.cost.append((-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))))

        return self

    def predict(self, x: np.matrix) -> np.ndarray:
        """
        Return class label after unit step
        """
        return np.where(self.activation(self.net_input(x)) >= 0.5, 1, 0)

    def activation(self, z):
        """
        Compute logistic sigmoid activation
        """
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
