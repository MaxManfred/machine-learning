import unittest

import numpy as np
from sklearn.cluster import KMeans

from ml.clustering.clustered_data_generator import ClusteredDataGenerator
from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils


class ClusteringQualityTest(unittest.TestCase):

    def setUp(self):
        # load data
        data_generator = ClusteredDataGenerator()
        self.x, _ = data_generator.create_data()

    def tearDown(self):
        return

    def test_elbow_method(self):
        # To quantify the quality of clustering, we need to use intrinsic metrics—such as the within-cluster
        # Sum of Squared Errors (SSE), which is sometimes also called cluster inertia or distortion to compare the
        # performance of different k-means clusterings.
        # Conveniently, we don't need to compute the within-cluster SSE explicitly when we are using scikit-learn, as it
        # is already accessible via the inertia_ attribute after fitting a KMeans model:
        #     >>> print('Distortion: %.2f' % km.inertia_)
        #     Distortion: 72.48
        # Based on the within-cluster SSE, we can use a graphical tool, the so-called elbow method, to estimate the
        # optimal number of clusters k for a given task. Intuitively, we can say that, if k increases, the distortion
        # will decrease. This is because the samples will be closer to the centroids they are assigned to.
        # The idea behind the elbow method is to identify the value of k where the distortion begins to increase most
        # rapidly

        distortions = []
        k_range = range(1, 11)
        for i in k_range:
            km = KMeans(
                # number of clusters
                n_clusters=i,
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

            km.fit(self.x)

            distortions.append(km.inertia_)

        data = np.column_stack((k_range, distortions))
        Plotter.plot_data(data, x_label='Number of clusters',
                          y_label='Distorsion', title='Elbow method for optimal number of clusters',
                          image_file_path=FilesystemUtils.get_test_resources_plot_file_name(
                              'clustering/ElbowMethodForKMeansClustering.png'))
