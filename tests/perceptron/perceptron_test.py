import unittest

from ml.common.data.data_reader import IrisDataReader
from ml.common.plot.plotter import Plotter
from ml.perceptron import Perceptron


class PerceptronTest(unittest.TestCase):

    def setUp(self):
        # load subset of Iris data
        iris_data_reader = IrisDataReader('../../resources/iris.data')
        self.x, self.y = iris_data_reader.get_data()

        # plotter data and save it to file
        Plotter.plot_data_set(self.x, '../../resources/images/Adaline-Training-Set.png')

    def tearDown(self):
        return

    def test_perceptron(self):
        # train the perceptron model
        perceptron = Perceptron(learning_rate=0.1, num_epochs=10)
        perceptron.fit(self.x, self.y)

        # plot learning
        curve = {
            'cost_length': len(perceptron.cost),
            'cost': perceptron.cost,
            'marker': 'o',
            'x_label': 'Epochs',
            'y_label': 'Number of updates',
            'title': 'Perceptron - Learning rate 0.1'
        }
        Plotter.plot_learning_curve(curve, '../../resources/images/Perceptron-Learning-Curve.png')

        # plot decision boundary
        diagram_options = {
            'x_label': 'sepal length [cm]',
            'y_label': 'petal length [cm]',
            'legend': 'upper left'
        }
        Plotter.plot_decision_boundary(self.x, self.y, classifier=perceptron, diagram_options=diagram_options,
                                       image_file_path='../../resources/images/Perceptron-Decision-Boundary.png')


if __name__ == '__main__':
    unittest.main()
