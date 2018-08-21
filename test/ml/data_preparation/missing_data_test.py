import unittest

from ml.data_preparation.missing_data import MissingData


class MissingDataTest(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        return

    @staticmethod
    def test_missing_data():
        missing_data = MissingData()
        missing_data.load_sample_csv_data()

        # replace missing values using the mean along the column axis (0)
        missing_data.get_imputed_data(strategy='mean', axis=0)

        # replace missing values using the mean along the row axis (1)
        missing_data.get_imputed_data(strategy='mean', axis=1)

        # replace missing values using the median along the column axis (0)
        missing_data.get_imputed_data(strategy='median', axis=0)

        # replace missing values using the median along the row axis (1)
        missing_data.get_imputed_data(strategy='median', axis=1)

        # replace missing values using the most frequent value along the column axis (0)
        missing_data.get_imputed_data(strategy='most_frequent', axis=0)

        # replace missing values using the most frequent value along the row axis (1)
        missing_data.get_imputed_data(strategy='most_frequent', axis=1)


if __name__ == '__main__':
    unittest.main()
