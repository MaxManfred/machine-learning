import array

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy import interp
from sklearn.metrics import auc

from ml.common.classification.classifier import Classifier


class Plotter(object):

    def __init__(self) -> None:
        """
        Constructor
        """

    @staticmethod
    def plot_data(data: np.matrix, color: str = 'blue', marker: str = 'o', title: str = '',
                  x_label: str = '', y_label: str = '', image_file_path: str = None, resolution: int = 300) -> None:
        plt.plot(data[:, 0], data[:, 1], c=color, marker=marker)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_scattered_data(data: np.matrix, color: str = 'white', marker: str = 'o', marker_size: int = 50,
                            edge_color: str = 'black', title: str = '', image_file_path: str = None, resolution: int = 300) -> None:
        plt.scatter(data[:, 0], data[:, 1], c=color, marker=marker, edgecolor=edge_color, s=marker_size)

        plt.grid()

        plt.title(title)

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

    @staticmethod
    def plot_multiple_scattered_data(data: [], centroids: {}, title: str, image_file_path: str = None,
                                     resolution: int = 300) -> None:
        # data is a dictionary list
        for d in data:
            plt.scatter(d['x'][:, 0], d['x'][:, 1], c=d['color'], marker=d['marker'], edgecolor=d['edge_color'],
                        s=d['marker_size'], label=d['label'])

        plt.scatter(centroids['x'][:, 0], centroids['x'][:, 1], c=centroids['color'], marker=centroids['marker'],
                    edgecolor=centroids['edge_color'], s=centroids['marker_size'], label=centroids['label'])

        plt.legend(scatterpoints=1)

        plt.grid()

        plt.title(title)

        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution)

        plt.show()

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

    @staticmethod
    def plot_feature_importance(feature_number: int, descending_importance_list: [], descending_names_list: [],
                                image_file_path: str = None,
                                resolution: int = 300):
        plt.title('Feature Importance')
        plt.bar(range(feature_number), descending_importance_list, align='center')
        plt.xticks(range(feature_number), descending_names_list, rotation=90)
        plt.xlim([-1, feature_number])
        plt.tight_layout()

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_performance_curves(train_sizes: np.array, train_scores: np.array, test_scores: np.array,
                                image_file_path: str = None, resolution: int = 300):
        # We simply calculated the average accuracies from the returned cross-validated training and test scores for
        # the different sizes of the training set, which we plotted using Matplotlib's plot function.
        # Furthermore, we added the standard deviation of the average accuracy to the plot using the fill_between
        # function to indicate the variance of the estimate.

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.title('Learning curves')
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

        plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                 label='validation accuracy')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

        plt.grid()

        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')

        plt.legend(loc='lower right')

        plt.ylim([0.8, 1.0])

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_validation_curves(param_range: np.array, train_scores: np.array, test_scores: np.array,
                               image_file_path: str = None, resolution: int = 300):
        # The validation_curve function uses stratified k-fold cross-validation by default to estimate the performance
        # of the classifier.
        # Similar to the learning curve, we plotted the average training and cross-validation accuracies and the
        # corresponding standard deviations.
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.title('Validation curves')
        plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

        plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5,
                 label='validation accuracy')
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

        plt.grid()

        plt.xscale('log')

        plt.legend(loc='lower right')

        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')

        plt.ylim([0.8, 1.03])

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.array, image_file_path: str = None, resolution: int = 300):
        # We can map this information onto the confusion matrix illustration using Matplotlib matshow function

        plt.title('Confusion matrix')
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center')

        plt.xlabel('predicted label')
        plt.ylabel('true label')

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_roc_auc(false_positive_rates: np.array, true_positive_rates: np.array, roc_auc_values: np.array,
                     image_file_path: str = None, resolution: int = 300):
        fig = plt.figure(figsize=(7, 5))
        plt.title('ROC AUC')

        # used to compute mean ROC and AUC
        tpr_mean = 0.0
        evenly_spaced_fpr = np.linspace(0, 1, 100)

        for i in range(0, len(false_positive_rates)):
            fpr = false_positive_rates[i]
            tpr = true_positive_rates[i]
            auc_value = roc_auc_values[i]
            # plot current ROC curve (for the current cv fold)
            plt.plot(fpr, tpr, label='ROC fold {:,} (area = {:.2f})'.format(i + 1, auc_value))

            # update values for mean ROC and AUC
            tpr_mean += interp(evenly_spaced_fpr, fpr, tpr)
            tpr_mean[0] = 0.0

        # normalize values for mean ROC and AUC
        tpr_mean /= len(false_positive_rates)
        tpr_mean[-1] = 1.0

        # compute mean AUC
        auc_mean = auc(evenly_spaced_fpr, tpr_mean)

        # plot mean ROC curve
        plt.plot(evenly_spaced_fpr, tpr_mean, 'k--', label='mean ROC (area = {:.2f})'.format(auc_mean), lw=2)

        # plot random guessing curve (45 degrees)
        plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')

        # plot perfect performance curve)
        plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', label='perfect performance')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.legend(loc="lower right")

        plt.savefig(image_file_path, dpi=resolution, bbox_inches='tight')

        plt.show()
