import unittest

import numpy as np
from sklearn.cluster import KMeans

from ml.clustering.clustered_data_generator import ClusteredDataGenerator
from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils


class CategoricalDataTest(unittest.TestCase):

    def setUp(self):
        # load data
        data_generator = ClusteredDataGenerator()
        self.x, _ = data_generator.create_data()

    def tearDown(self):
        return

    def test_plot_sample_clustered_data(self):
        Plotter.plot_scattered_data(
            self.x,
            title='Sample clustered data',
            image_file_path=FilesystemUtils.get_test_resources_plot_file_name('clustering/SampleClusteredData.png')
        )

    def test_k_means_with_random_initialization(self):
        km = KMeans(
            # number of clusters
            n_clusters=3,
            # initialization method
            init='random',
            # number of different experiments run
            n_init=10,
            # maximum number of iteration
            max_iter=300,
            # change within cluster minimum threshold: below it, the training process is stopped
            tol=1e-04,
            # initialization seed
            random_state=42
        )

        predictions = km.fit_predict(self.x)

        self.plot_predictions(predictions, km.cluster_centers_,
                              title='Sample k-means clusters with random initialization',
                              diagram_file_name='SampleKMeansClustersWithRandomInitialization.png')

    def test_k_means_with_k_means_plus_plus_initialization(self):
        km = KMeans(
            # number of clusters
            n_clusters=3,
            # initialization method
            init='k-means++',
            # number of different experiments run
            n_init=10,
            # maximum number of iteration
            max_iter=300,
            # change within cluster minimum threshold: below it, the training process is stopped
            tol=1e-04,
            # initialization seed
            random_state=42
        )

        predictions = km.fit_predict(self.x)

        self.plot_predictions(predictions, km.cluster_centers_,
                              title='Sample k-means clusters with k-means++ initialization',
                              diagram_file_name='SampleKMeansClustersWithKMeans++Initialization.png')

    def plot_predictions(self, predictions: np.matrix, centroids: np.matrix, title: str, diagram_file_name: str):
        data = [
            {
                'x': self.x[predictions == 0, :],
                'color': 'lightgreen',
                'marker': 's',
                'marker_size': 50,
                'edge_color': 'black',
                'label': 'cluster 1'
            },
            {
                'x': self.x[predictions == 1, :],
                'color': 'orange',
                'marker': 's',
                'marker_size': 50,
                'edge_color': 'black',
                'label': 'cluster 2'
            },
            {
                'x': self.x[predictions == 2, :],
                'color': 'lightblue',
                'marker': 's',
                'marker_size': 50,
                'edge_color': 'black',
                'label': 'cluster 3'
            }
        ]

        centroids = {
            'x': centroids,
            'color': 'red',
            'marker': '*',
            'marker_size': 250,
            'edge_color': 'black',
            'label': 'centroids'
        }

        Plotter.plot_multiple_scattered_data(
            data, centroids, title,
            image_file_path=FilesystemUtils.get_test_resources_plot_file_name('clustering/' + diagram_file_name)
        )
