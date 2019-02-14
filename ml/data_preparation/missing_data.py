from io import StringIO

import pandas as pd
from sklearn.preprocessing import Imputer


class MissingData(object):

    def __init__(self):
        self.csv_data: str = None
        self.df: pd.DataFrame = None

    def load_sample_csv_data(self):
        self.csv_data = '''
            A,B,C,D 
            1.0,2.0,3.0,4.0 
            5.0,6.0,,8.0 
            10.0,11.0,12.0,
        '''

        # load csv data into a pandas dataframe
        self.df = pd.read_csv(StringIO(self.csv_data))
        print('\nLoaded DataFrame: ')
        print(self.df)

        print('\nSumming up null values per column: ')
        print(self.df.isnull().sum())

        print('\nAccessing underlying NumPy array through "value" attribute:')
        print(self.df.values)

        print('\nDrop rows that have at least one NaN in any column:')
        print(self.df.dropna(axis=0))

        print('\nDrop columns that have at least one NaN in any row:')
        print(self.df.dropna(axis=1))

        print('\nDrop rows where all columns are NaN:')
        print(self.df.dropna(how='all'))

        print('\nDrop rows that have less than 4 non null values: ')
        print(self.df.dropna(thresh=4))

        print('\nDrop rows  where NaN appear in specific columns (here: C): ')
        print(self.df.dropna(subset=['C']))

        print('\n')

    def get_imputed_data(self, strategy: str = 'mean', axis: int = 0):
        print('\nCreating imputed data with strategy {0} and axis {1}'.format(strategy,
                                                                              'column' if axis == 0 else 'row' if axis == 1 else ''))
        imputer = Imputer(missing_values='NaN', strategy=strategy, axis=axis)
        imputer = imputer.fit(self.df.values)

        imputed_data = imputer.transform(self.df.values)
        print(imputed_data)

        print('\n')
