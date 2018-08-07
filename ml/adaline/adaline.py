import numpy as np
from numpy.random.mtrand import RandomState

from ml.common.classification import Classifier


class AdalineBGD(Classifier):
    """
    Adaptive Linear Neurons with Batch Gradient Descend
    """

    def __init__(self, learning_rate: float = 0.01, num_epochs: int = 50, weight_init_seed: int = 1) -> None:
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_seed = weight_init_seed
        self.cost = []
        self.weights = None

    def activation(self, x):
        """Compute linear activation"""
        return x

    def fit(self, x: np.matrix, y: np.matrix):
        random_number_generator: RandomState = RandomState(self.weight_init_seed)
        self.weights = random_number_generator.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])

        for i in range(self.num_epochs):
            print('Training epoch ' + str(i + 1))

            # Please note that the "activation" method has no effect in the code since it is simply an identity
            # function.
            # We could write output = self.net_input(x) directly instead.
            # The purpose of the activation is more conceptual, i.e., in the case of logistic regression (as we will see
            # later),
            # we could change it to a sigmoid function to implement a logistic regression classifier.
            output = self.activation(self.net_input(x))
            errors = (y - output)

            self.weights[1:] += self.learning_rate * x.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

            self.cost.append((errors ** 2).sum() / 2.0)

        return self

    def predict(self, x: np.matrix) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)


class AdalineSGD(Classifier):
    """
    Adaptive Linear Neurons with Stochastic Gradient Descend
    """

    def __init__(self, learning_rate: float = 0.01, num_epochs: int = 50, weight_init_seed: int = 1, shuffle=True) \
            -> None:
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_init_seed = weight_init_seed
        self.shuffle = shuffle

        self.cost = []
        self.weights = None
        self.weights_initialized = False
        self.random_number_generator: RandomState = RandomState(self.weight_init_seed)

    def activation(self, x):
        """Compute linear activation"""
        return x

    def fit(self, x: np.matrix, y: np.matrix):
        # Initialize weights to small random numbers
        self._initialize_weights(x.shape[1])

        self.cost = []
        for i in range(self.num_epochs):
            print('Training epoch ' + str(i + 1))

            if self.shuffle:
                x, y = self._shuffle(x, y)

            c = []
            for xi, target in zip(x, y):
                c.append(self._update_weights(xi, target))
            average_cost = sum(c) / len(y)
            self.cost.append(average_cost)

        return self

    def partial_fit(self, x, y):
        """
        Fit training data without reinitializing the weights
        """
        if not self.weights_initialized:
            self._initialize_weights(x.shape[1])

        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)

        return self

    def predict(self, x: np.matrix) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(x)) >= 0.0, 1, -1)

    def _initialize_weights(self, number_of_features: int):
        """
        Initialize weights to small random numbers
        """
        self.weights = self.random_number_generator.normal(loc=0.0, scale=0.01, size=1 + number_of_features)
        self.weights_initialized = True

    def _shuffle(self, x, y):
        """
        Shuffle training data
        """
        r = self.random_number_generator.permutation(len(y))

        return x[r], y[r]

    def _update_weights(self, xi, target):
        """
        Apply Adaline learning rule to update the weights
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.weights[1:] += self.learning_rate * xi.dot(error)
        self.weights[0] += self.learning_rate * error
        c = 0.5 * error**2

        return c
