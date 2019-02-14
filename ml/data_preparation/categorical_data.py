import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class CategoricalData(object):

    def __init__(self):
        self.csv_data: str = None
        self.df: pd.DataFrame = None

    def load_sample_data_frame(self):
        self.df = pd.DataFrame(
            [
                ['green', 'M', 10.1, 'class1'],
                ['red', 'L', 13.5, 'class2'],
                ['blue', 'XL', 15.3, 'class1']
            ]
        )
        self.df.columns = ['color', 'size', 'price', 'classlabel']

        print('\nLoaded DataFrame: ')
        print(self.df)

    def map_ordinal_feature(self):
        # Mapping ordinal feature size through a mapping dictionary
        self.size_mapping = {'XL': 3, 'L': 2, 'M': 1}

        self.df['size'] = self.df['size'].map(self.size_mapping)

        print('\nMapped ordinal feature size: ')
        print(self.df)

    def reverse_map_ordinal_feature(self):
        inv_size_mapping = {v: k for k, v in self.size_mapping.items()}
        self.df['size'] = self.df['size'].map(inv_size_mapping)

        print('\nUnmapped ordinal feature size: ')
        print(self.df)

    def encode_class_labels(self):
        # Mapping class labels through a mapping dictionary
        self.class_mapping = {label: idx for idx, label in enumerate(np.unique(self.df['classlabel']))}
        self.df['classlabel'] = self.df['classlabel'].map(self.class_mapping)

        print('\nEncoded class labels: ')
        print(self.df)

    def decode_class_labels(self):
        inv_class_mapping = {v: k for k, v in self.class_mapping.items()}
        self.df['classlabel'] = self.df['classlabel'].map(inv_class_mapping)

        print('\nDecoded class labels: ')
        print(self.df)

    def encode_decode_class_labels_with_scikit(self):
        class_le = LabelEncoder()
        self.df['classlabel'] = class_le.fit_transform(self.df['classlabel'].values)

        print('\nEncoded class labels with scikit: ')
        print(self.df)

        self.df['classlabel'] = class_le.inverse_transform(self.df['classlabel'].values)

        print('\nDecoded class labels with scikit: ')
        print(self.df)

    def one_hot_encode_nominal_feature_with_scikit(self):
        X = self.df[['color', 'size', 'price']].values
        color_le = LabelEncoder()
        X[:, 0] = color_le.fit_transform(X[:, 0])
        print(X)

        ohe = OneHotEncoder(categorical_features=[0])
        self.df = ohe.fit_transform(X).toarray()

        print('\nOne-hot encoded data set with scikit: ')
        print(self.df)
