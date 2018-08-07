import unittest

import numpy as np

from ml.adaline.adaline import AdalineBGD, AdalineSGD
from ml.common.data.data_reader import IrisDataReader
from ml.common.plot.plotter import Plotter


class AdalineTest(unittest.TestCase):

    def setUp(self):
        # load subset of Iris data
        iris_data_reader = IrisDataReader('../../resources/iris.data')
        self.x, self.y = iris_data_reader.get_data()

        # plotter data and save it to file
        Plotter.plot_data_set(self.x, '../../resources/images/Adaline-Training-Set.png')

    def tearDown(self):
        return

    def test_adaline(self):
        # train the first  with bigger learning ratio
        adaline1 = AdalineBGD(learning_rate=0.01, num_epochs=30)
        adaline1.fit(self.x, self.y)

        # train the second adaline model with smaller learning ration
        adaline2 = AdalineBGD(learning_rate=0.0001, num_epochs=30)
        adaline2.fit(self.x, self.y)

        # plot multiple learning curves for both adaline trained models
        curves = [
            {
                'cost_length': len(adaline1.cost),
                'cost': np.log10(adaline1.cost),
                'marker': 'o',
                'x_label': 'Epochs',
                'y_label': 'log(Sum-squared-error)',
                'title': 'Adaline - Learning rate 0.1'
            },
            {
                'cost_length': len(adaline2.cost),
                'cost': np.log10(adaline2.cost),
                'marker': 'o',
                'x_label': 'Epochs',
                'y_label': 'log(Sum-squared-error)',
                'title': 'Adaline - Learning rate 0.0001'
            }
        ]
        Plotter.plot_multiple_learning_curves(curves,
                                              image_file_path='../../resources/images/AdalineBGD-Learning-Curves.png')

        # plot decision boundary for divergent model (adaline 1)
        Plotter.plot_decision_boundary(self.x, self.y, classifier=adaline1,
                                       diagram_options={'x_label': 'sepal length [cm]',
                                                        'y_label': 'petal length [cm]',
                                                        'legend': 'upper left'},
                                       image_file_path='../../resources/images/AdalineBGD-Decision-Boundary-Divergenr.png')

        # plot decision boundary for convergent model (adaline 2)
        Plotter.plot_decision_boundary(self.x, self.y, classifier=adaline2,
                                       diagram_options={'x_label': 'sepal length [cm]',
                                                        'y_label': 'petal length [cm]',
                                                        'legend': 'upper left'},
                                       image_file_path='../../resources/images/AdalineBGD-Decision-Boundary-Convergent.png'
                                       )

    def test_adaline_with_standardized_features(self):
        # standardize features
        x_std: np.matrix = np.copy(self.x)
        x_std[:, 0] = (self.x[:, 0] - self.x[:, 0].mean()) / self.x[:, 0].std()
        x_std[:, 1] = (self.x[:, 1] - self.x[:, 1].mean()) / self.x[:, 1].std()

        # plotter data and save it to file
        Plotter.plot_data_set(x_std, '../../resources/images/AdalineBGD-Standardized-Training-Set.png')

        # train adaline on standardized features with a small number of epochs
        adaline = AdalineBGD(learning_rate=0.01, num_epochs=10)
        adaline.fit(x_std, self.y)

        # plot learning curve
        curve = {
            'cost_length': len(adaline.cost),
            'cost': np.log10(adaline.cost),
            'marker': 'o',
            'x_label': 'Epochs',
            'y_label': 'log(Sum-squared-error)',
            'title': 'Adaline - Learning rate 0.01'
        }
        Plotter.plot_learning_curve(curve, '../../resources/images/Adaline-Learning-Curve-Standardized-Features.png')

        # plot decision boundary
        Plotter.plot_decision_boundary(x_std, self.y, classifier=adaline,
                                       diagram_options={'x_label': 'sepal length [cm]',
                                                        'y_label': 'petal length [cm]',
                                                        'legend': 'upper left'},
                                       image_file_path='../../resources/images/Adaline-Decision-Boundary-Standardized'
                                                       '-Features.png')

    def test_adaline_with_stochastic_update(self):
        # standardize features
        x_std: np.matrix = np.copy(self.x)
        x_std[:, 0] = (self.x[:, 0] - self.x[:, 0].mean()) / self.x[:, 0].std()
        x_std[:, 1] = (self.x[:, 1] - self.x[:, 1].mean()) / self.x[:, 1].std()

        # plotter data and save it to file
        Plotter.plot_data_set(x_std, '../../resources/images/AdalineSGD-Standardized-Training-Set.png')

        # train adaline on standardized features with a small number of epochs
        adaline = AdalineSGD(learning_rate=0.01, num_epochs=15)
        adaline.fit(x_std, self.y)

        # plot learning curve
        curve = {
            'cost_length': len(adaline.cost),
            'cost': adaline.cost,
            'marker': 'o',
            'x_label': 'Epochs',
            'y_label': 'log(Sum-squared-error)',
            'title': 'Adaline - Learning rate 0.01'
        }
        Plotter.plot_learning_curve(curve, '../../resources/images/AdalineSGD-Learning-Curve-Standardized-Features.png')

        # plot decision boundary
        Plotter.plot_decision_boundary(x_std, self.y, classifier=adaline,
                                       diagram_options={'x_label': 'sepal length [cm]',
                                                        'y_label': 'petal length [cm]',
                                                        'legend': 'upper left'},
                                       image_file_path='../../resources/images/AdalineSGD-Decision-Boundary-Standardized'
                                                       '-Features.png')

        adaline.partial_fit(x_std[0, :], self.y[0])


if __name__ == '__main__':
    unittest.main()
