import unittest

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
            image_file_path=FilesystemUtils.get_test_resources_plot_file_name('clustering/SampleClusteredData.png')
        )

    def test_k_means_with_random_initialization(self):
        km = KMeans(
            # number of clusters
            n_clusters=3,
            # initialization mathod
            init='random',
            # number of different experiments run
            n_init=10,
            # maximum number of iteration
            max_iter=300,
            # change within cluster inimum threshold: below it, the training process is stopped
            tol=1e-04,
            # initialization seed
            random_state=42
        )

        predictions = km.fit_predict(self.x)

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
            'x': km.cluster_centers_,
            'color': 'red',
            'marker': '*',
            'marker_size': 250,
            'edge_color': 'black',
            'label': 'centroids'
        }

        Plotter.plot_multiple_scattered_data(
            data, centroids,
            image_file_path=FilesystemUtils.get_test_resources_plot_file_name('clustering/SampleKMeansClusters.png')
        )
