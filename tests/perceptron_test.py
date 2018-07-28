import unittest

from ml.common.data.data_reader import IrisDataReader
from ml.common.plot.plotter import Plotter
from ml.perceptron.perceptron import Perceptron


class PerceptronTest(unittest.TestCase):

    def set_up(self):
        return

    def tear_down(self):
        return

    def test_perceptron(self):
        # load subset of Iris data
        iris_data_reader = IrisDataReader('../resources/iris.data')
        X, Y = iris_data_reader.get_data()

        # plotter data and save it to file
        Plotter.plot_data_set(X, '../resources/images/Perceptron-Training-Set.png')

        # train the perceptron model
        perceptron = Perceptron(learning_rate=0.1, num_epochs=10)
        perceptron.train(X, Y)

        # plot learning curve
        Plotter.plot_learning_curve(perceptron.errors, '../resources/images/Perceptron-Learning-Curve.png')

        # plot decision boundary
        Plotter.plot_decision_boundary(X, Y, classifier=perceptron, x_label='sepal length [cm]',
                                       y_label='petal length [cm]', legend='upper left',
                                       image_file_path='../resources/images/Perceptron-Decision-Boundary.png')


if __name__ == '__main__':
    unittest.main()
