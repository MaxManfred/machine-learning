import unittest

from ml.data_preparation.categorical_data import CategoricalData


class CategoricalDataTest(unittest.TestCase):

    def setUp(self):
        self.categorical_data = CategoricalData()
        # load data
        self.categorical_data.load_sample_data_frame()

    def tearDown(self):
        return

    def test_map_unmap_ordinal_feature(self):
        # Mapping ordinal feature size through a mapping dictionary
        self.categorical_data.map_ordinal_feature()

        # Unmapping ordinal feature size through a mapping dictionary
        self.categorical_data.reverse_map_ordinal_feature()

    def test_encode_decode_class_label(self):
        # Encoding class labels through a mapping dictionary
        self.categorical_data.encode_class_labels()

        # Decoding class labels through a mapping dictionary
        self.categorical_data.decode_class_labels()

    def test_encode_decode_class_label_with_scikit(self):
        # Encoding and decoding class labels through scikit LabelEncoder
        self.categorical_data.encode_decode_class_labels_with_scikit()

        # Note: The deprecation warning shown when running this test is due to an implementation detail in scikit-learn.
        # It was already addressed in a pull request (https://github.com/scikit-learn/scikit-learn/pull/9816),
        # and the patch will be released with the next version of scikit-learn (i.e., v. 0.20.0).

    def test_one_hot_encoding_with_scikit(self):
        self.categorical_data.map_ordinal_feature()
        self.categorical_data.one_hot_encode_nominal_feature_with_scikit()


if __name__ == '__main__':
    unittest.main()
