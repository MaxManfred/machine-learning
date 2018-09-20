from numpy import amax, amin, mean, std
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from test.ml.common.wine_dataset_test import WineDatasetTest


class NormalizationStandardizationTest(WineDatasetTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_normalize_features_with_scikit': cls.load_wine_data_set,
            'test_standardize_features_with_scikit': cls.load_wine_data_set
        }

    def test_normalize_features_with_scikit(self):
        # first, split the entire dataset into the training and testing subsets just for pretending this is a real
        # training scenario
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # create normalizer
        min_max_scaler = MinMaxScaler()

        # call fit to compute the minimum and maximum to be used for later scaling and then transform for actual scaling
        min_max_scaler.fit(X_train)
        X_train_normalized = min_max_scaler.transform(X_train)
        assert amin(X_train_normalized) >= 0
        assert amax(X_train_normalized) <= 1

        # call the fit_transform shortcut
        X_test_normalized = min_max_scaler.fit_transform(X_test)
        assert amin(X_test_normalized) >= 0
        assert amax(X_test_normalized) <= 1

    def test_standardize_features_with_scikit(self):
        # first, split the entire dataset into the training and testing subsets just for pretending this is a real
        # training scenario
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        # create standizer
        standard_scaler = StandardScaler()

        # call fit to compute the minimum and maximum to be used for later scaling and then transform for actual scaling
        standard_scaler.fit(X_train)
        X_train_standardized = standard_scaler.transform(X_train)
        # notice mean could be very small but non-zero, so we round down
        assert int(mean(X_train_standardized)) == 0
        assert int(std(X_train_standardized)) == 1

        # call the fit_transform shortcut
        X_test_standardized = standard_scaler.fit_transform(X_test)
        assert int(mean(X_test_standardized)) == 0
        assert int(std(X_test_standardized)) == 1
