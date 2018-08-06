import unittest

import numpy as np

from ml.adaline.adaline import AdalineGD
from ml.common.data.data_reader import IrisDataReader
from ml.common.plot.plotter import Plotter


class AdalineTest(unittest.TestCase):

    def set_up(self):
        return

    def tear_down(self):
        return

    def test_adaline(self):
        # load subset of Iris data
        iris_data_reader = IrisDataReader('../../resources/iris.data')
        x, y = iris_data_reader.get_data()

        # plotter data and save it to file
        Plotter.plot_data_set(x, '../../resources/images/Adaline-Training-Set.png')

        # train the first  with bigger learning ratio
        adaline1 = AdalineGD(learning_rate=0.01, num_epochs=30)
        adaline1.train(x, y)

        # train the second adaline model with smaller learning ration
        adaline2 = AdalineGD(learning_rate=0.0001, num_epochs=30)
        adaline2.train(x, y)

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
                                              image_file_path='../../resources/images/Adaline-Learning-Curves.png')

        # plot decision boundary for divergent model (adaline 1)
        Plotter.plot_decision_boundary(x, y, classifier=adaline1, diagram_options={'x_label': 'sepal length [cm]',
                                                                                   'y_label': 'petal length [cm]',
                                                                                   'legend': 'upper left'},
                                       image_file_path='../../resources/images/Adaline-Decision-Boundary-Divergenr.png')

        # plot decision boundary for convergent model (adaline 2)
        Plotter.plot_decision_boundary(x, y, classifier=adaline2, diagram_options={'x_label': 'sepal length [cm]',
                                                                                   'y_label': 'petal length [cm]',
                                                                                   'legend': 'upper left'},
                                       image_file_path='../../resources/images/Adaline-Decision-Boundary-Convergent.png')

    def test_adaline_with_standardized_features(self):
        # load subset of Iris data
        iris_data_reader = IrisDataReader('../../resources/iris.data')
        x, y = iris_data_reader.get_data()

        # standardize features
        x_std: np.matrix = np.copy(x)
        x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
        x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

        # plotter data and save it to file
        Plotter.plot_data_set(x, '../../resources/images/Adaline-Training-Set.png')
        Plotter.plot_data_set(x_std, '../../resources/images/Adaline-Standardized-Training-Set.png')

        # train adaline on standardized features with a small number of epochs
        adaline = AdalineGD(learning_rate=0.01, num_epochs=10)
        adaline.train(x_std, y)

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
        Plotter.plot_decision_boundary(x_std, y, classifier=adaline, diagram_options={'x_label': 'sepal length [cm]',
                                                                                      'y_label': 'petal length [cm]',
                                                                                      'legend': 'upper left'},
                                       image_file_path='../../resources/images/Adaline-Decision-Boundary-Standardized'
                                                       '-Features.png')


#


#         ada = AdalineGD(n_iter=15, eta=0.01)
# ada.fit(X_std, y)
#
# plot_decision_regions(X_std, y, classifier=ada)
# plt.title('Adaline - Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# # plt.savefig('images/02_14_1.png', dpi=300)
# plt.show()
#
# plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')
#
#
# # plt.savefig('images/02_14_2.png', dpi=300)
# plt.show()


if __name__ == '__main__':
    unittest.main()
