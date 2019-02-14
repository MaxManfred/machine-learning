from itertools import combinations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# The idea behind the SBS algorithm is quite simple: SBS sequentially removes features from the full feature subset
# until the new feature subspace contains the desired number of features. In order to determine which feature is to be
# removed at each stage, we need to define the criterion function J that we want to minimize. The criterion calculated
# by the criterion function can simply be the difference in performance of the classifier before and after the removal
# of a particular feature. Then, the feature to be removed at each stage can simply be defined as the feature that
# maximizes  this criterion; or in more intuitive terms, at each stage we eliminate the feature that causes the least
# performance loss after removal.
class SequentialFeatureSelection(object):

    def __init__(self, estimator, selected_features_number, scoring_metric=accuracy_score, test_size=0.25,
                 random_state=42):
        self.estimator = clone(estimator)
        self.selected_features_number = selected_features_number
        self.scoring_metric = scoring_metric
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        # The k_features parameter specifies the desired number of features we want to return.
        # By default, we use the accuracy_score from scikit-learn to evaluate the performance of a model
        # (an estimator for classification) on the feature subsets.
        # Inside the while loop of the fit method, the feature subsets created by the itertools.combination function are
        # evaluated and reduced until the feature subset has the desired dimensionality. In each iteration, the accuracy
        # score of the best subset is collected in a list, self.scores_, based on the internally created test dataset
        # X_test. We will use those scores later to evaluate the results. The column indices of the final feature subset
        # are assigned to self.indices_, which we can use via the transform method to return a new data array with the
        # selected feature columns. Note that, instead of calculating the criterion explicitly inside the fit method, we
        # simply removed the feature that is not contained in the best performing feature subset.

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size,
                                                            random_state=self.random_state)

        dim = x_train.shape[1]

        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]

        score = self.compute_model_score(x_train, y_train, x_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.selected_features_number:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self.compute_model_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

    def transform(self, x):
        return x[:, self.indices_]

    def compute_model_score(self, x_train, y_train, x_test, y_test, selected_feature_indices):
        self.estimator.fit(x_train[:, selected_feature_indices], y_train)
        y_pred = self.estimator.predict(x_test[:, selected_feature_indices])
        score = self.scoring_metric(y_test, y_pred)

        return score
