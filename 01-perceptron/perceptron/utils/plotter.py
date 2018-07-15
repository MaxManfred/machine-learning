import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron.classifier import Classifier


class Plotter(object):

    def __init__(self) -> None:
        """
        Constructor
        """

    def plot_data_set(self, X: np.matrix, image_file_path: str = None, resolution: int = 300) -> None:
        plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')

        plt.savefig(image_file_path, dpi=resolution)
        plt.show()

    def plot_learning_curve(self, errors: [int] = None, image_file_path: str = None, resolution: int = 300):
        plt.plot(range(1, len(errors) + 1), errors, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')

        plt.savefig(image_file_path, dpi=resolution)
        plt.show()

    def plot_decision_boundary(self, X, Y, grid_resolution: float = 0.02, classifier: Classifier = None, x_label: str = None,
                               y_label: str = None, legend: str = None, image_file_path: str = None, resolution: int = 300):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(Y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, grid_resolution), np.arange(x2_min, x2_max, grid_resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(Y)):
            plt.scatter(
                x=X[Y == cl, 0],
                y=X[Y == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=cl,
                edgecolor='black'
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc=legend)

        plt.savefig(image_file_path, dpi=resolution)
        plt.show()