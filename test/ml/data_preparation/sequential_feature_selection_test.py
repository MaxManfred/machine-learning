from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ml.common.plot import Plotter
from ml.data_preparation.sequential_feature_selection import SequentialFeatureSelection
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.wine_dataset_test import WineDatasetTest


class SequentialFeatureSelectionTest(WineDatasetTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_sequential_feature_selection': cls.load_wine_data_set
        }

    def test_sequential_feature_selection(self):
        knn = KNeighborsClassifier(n_neighbors=5)

        # standardize features
        stdsc = StandardScaler()
        x_std = stdsc.fit_transform(self.x)

        # selecting features
        sbs = SequentialFeatureSelection(knn, selected_features_number=1)
        sbs.fit(x_std, self.y)

        Plotter.plot_accuracy_by_feature_number(
            sbs.subsets_, sbs.scores_,
            FilesystemUtils.get_test_resources_plot_file_name('sequential_feature_selection/AccuracyByFeatureNumber.png')
        )
