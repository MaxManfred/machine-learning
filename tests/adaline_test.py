import unittest
import numpy as np

from ml.adaline.adaline import AdalineGD
from ml.common.data.data_reader import IrisDataReader
from ml.common.plot.plotter import Plotter
import matplotlib.pyplot as plt


class AdalineTest(unittest.TestCase):

    def set_up(self):
        return

    def tear_down(self):
        return

    def test_padaline(self):
        # load subset of Iris data
        iris_data_reader = IrisDataReader('../resources/iris.data')
        X, Y = iris_data_reader.get_data()

        # plotter data and save it to file
        plotter = Plotter()
        plotter.plot_data_set(X, '../resources/images/01-adaline-A.png')

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        # train the perceptron model
        adaline1 = AdalineGD(learning_rate=0.1, num_epochs=10)
        adaline1.train(X, Y)

        ax[0].plot(range(1, len(adaline1.cost) + 1), np.log10(adaline1.cost), marker='o')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('log(Sum-squared-error)')
        ax[0].set_title('Adaline - Learning rate 0.01')

        adaline2 = AdalineGD(learning_rate=0.0001, num_epochs=10)
        adaline2.train(X, Y)

        ax[1].plot(range(1, len(adaline2.cost) + 1), np.log10(adaline2.cost), marker='o')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Sum-squared-error')
        ax[1].set_title('Adaline - Learning rate 0.0001')

        # plt.savefig('images/02_11.png', dpi=300)
        plt.show()

        # # plot learning curve
        # plotter.plot_learning_curve(adaline.cost, '../resources/images/01-adaline-B.png')
        #
        # # plot decision boundary
        # plotter.plot_decision_boundary(X, Y, classifier=adaline, x_label='sepal length [cm]',
        #                                y_label='petal length [cm]', legend='upper left',
        #                                image_file_path='../resources/images/01-adaline-C.png')


if __name__ == '__main__':
    unittest.main()
