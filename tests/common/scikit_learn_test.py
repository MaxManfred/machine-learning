import unittest

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.common.plot.plotter import Plotter


class ScikitLearnTest(unittest.TestCase):

    switcher = {}

    def load_iris_dataset(self):
        # Loading the Iris dataset from scikit-learn.
        # The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.
        iris = datasets.load_iris()
        x = iris.data[:, [2, 3]]
        y = iris.target
        print('Class labels:', np.unique(y))

        # plotter data and save it to file
        Plotter.plot_iris_data_set(x, '../../resources/images/Perceptron-ScikitLearn-Training-Set.png')

        # Splitting data into 70% training and 30% test data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
        print('Labels counts in y:', np.bincount(y))
        print('Labels counts in y_train:', np.bincount(y_train))
        print('Labels counts in y_test:', np.bincount(y_test))

        # Standardize features
        sc = StandardScaler()
        sc.fit(x_train)
        x_train_std = sc.transform(x_train)
        sc.fit(x_test)
        x_test_std = sc.transform(x_test)

        self.x_train = x_train_std
        self.y_train = y_train

        self.x_test = x_test_std
        self.y_test = y_test

    def setUp(self):
        # Execute custom loading data function specified in child test class, otherwise switch to the default loading iris
        # data function
        func = self.switcher.get(self._testMethodName, self.load_iris_dataset)
        if 'load_iris_dataset'.__eq__(func.__name__):
            func()
        else:
            func(self)

    def tearDown(self):
        return
