from sklearn.datasets import make_blobs


class ClusteredDataGenerator(object):

    def create_data(self):
        x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=42)

        return x, y
