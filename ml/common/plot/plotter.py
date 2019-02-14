import array

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ml.common.classification.classifier import Classifier


class Plotter(object):

    def __init__(self) -> None:
        """
        Constructor
        """

    @staticmethod
    def plot_iris_data_set(x: np.matrix, image_file_path: str = None, resolution: int = 300) -> None:
        plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
        plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')

        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_svm_nonlinear_data_set(x: np.matrix, y: np.matrix, image_file_path: str = None,
                                    resolution: int = 300) -> None:
        plt.scatter(x[y == 1, 0], x[y == 1, 1], c='b', marker='x', label='1')
        plt.scatter(x[y == -1, 0], x[y == -1, 1], c='r', marker='s', label='-1')

        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.xlabel('feature 1')
        plt.ylabel('feature 2')
        plt.legend(loc='best')

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_learning_curve(curve: {} = None, image_file_path: str = None, resolution: int = 300):
        plt.plot(range(1, curve['cost_length'] + 1), curve['cost'], marker=curve['marker'])
        plt.xlabel(curve['x_label'])
        plt.ylabel(curve['y_label'])
        plt.title(curve['title'])

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_multiple_learning_curves(curves: [{}] = None, figure_size: tuple = (10, 4), image_file_path: str = None,
                                      resolution: int = 300):
        fig, ax = plt.subplots(nrows=1, ncols=len(curves), figsize=figure_size)
        i: int = 0
        for curve in curves:
            ax[i].plot(range(1, curve['cost_length'] + 1), curve['cost'], marker=curve['marker'])
            ax[i].set_xlabel(curve['x_label'])
            ax[i].set_ylabel(curve['y_label'])
            ax[i].set_title(curve['title'])
            i += 1

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_decision_boundary(x: np.matrix, y: np.matrix, classifier: Classifier = None, diagram_options: dict = None,
                               image_file_path: object = None, resolution: object = 300) -> None:
        grid_resolution: float = diagram_options['grid_resolution'] if diagram_options.get(
            'grid_resolution') is not None \
            else 0.02
        x_label: str = diagram_options['x_label']
        y_label: str = diagram_options['y_label']
        legend: str = diagram_options['legend']

        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        color_map = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
        x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, grid_resolution), np.arange(x2_min, x2_max, grid_resolution))

        z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, z, alpha=0.3, cmap=color_map)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(
                x=x[y == cl, 0],
                y=x[y == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=cl,
                edgecolor='black'
            )

        # highlight test samples
        draw_test_samples: array[int] = diagram_options['draw_test_samples'] if diagram_options.get(
            'draw_test_samples') is not None else False

        if draw_test_samples:
            x_test, y_test = x[draw_test_samples, :], y[draw_test_samples]
            plt.scatter(x_test[:, 0], x_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100,
                        label='test set')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc=legend)

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_regularizatiion_curves(curve: {} = None, params: np.ndarray = None, weights: np.ndarray = None,
                                    image_file_path: str = None, resolution: int = 300):
        plt.plot(params, weights[:, 0], label='petal length')
        plt.plot(params, weights[:, 1], linestyle='--', label='petal width')

        plt.xlabel(curve['x_label'])
        plt.ylabel(curve['y_label'])
        plt.title(curve['title'])
        plt.legend(loc=curve['legend'])

        plt.xscale('log')

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_impurity_criteria(x_range: [], entropy: [], scaled_entropy: [], gini: [], misclassification_error: [],
                               image_file_path: str = None, resolution: int = 300):
        iterator = zip(
            [entropy, scaled_entropy, gini, misclassification_error],
            ['I(H)', '1/2 * I(H)', 'I(G)', 'I(E)'],
            ['-', '-', '--', '-.'],
            ['black', 'lightgray', 'red', 'green', 'cyan']
        )

        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)

        for i, lab, ls, c, in iterator:
            ax.plot(x_range, i, label=lab, linestyle=ls, lw=2, color=c)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, fancybox=True, shadow=False)

        ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
        ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
        plt.ylim([0, 1.1])
        plt.xlabel('p(i=1)')
        plt.ylabel('Impurity Index')

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_variable_feature_weights(weights: [], params: [], df_columns: [], image_file_path: str = None,
                                      resolution: int = 300):
        ax = plt.subplot(111)
        colors = [
            'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray',
            'indigo', 'orange'
        ]

        for column, color in zip(range(weights.shape[1]), colors):
            plt.plot(params, weights[:, column], label=df_columns[column + 1], color=color)

        plt.axhline(0, color='black', linestyle='--', linewidth=2)

        plt.xlim([10 ** (-5), 10 ** 5])

        plt.ylabel('weight coefficient')
        plt.xlabel('C')

        plt.xscale('log')
        plt.legend(loc='upper left')
        ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.02), ncol=1, fancybox=True)

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_accuracy_by_feature_number(feature_subsets: list, scores: list, image_file_path: str = None,
                                        resolution: int = 300):
        # plotting performance of feature subsets
        feature_number_list = [len(subset) for subset in feature_subsets]

        plt.plot(feature_number_list, scores, marker='o')
        plt.ylim([0.7, 1.02])
        plt.ylabel('Accuracy')
        plt.xlabel('Number of features')
        plt.grid()
        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()
