import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.common.plot import Plotter
from test.ml.common.filesystem_utils import FilesystemUtils
from test.ml.common.wine_dataset_test import WineDatasetTest


class FeatureImportanceTest(WineDatasetTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_feature_importance': cls.load_wine_data_set
        }

    def test_feature_importance(self):
        # split training and testing dataset
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.25, random_state=42)

        # standardize features
        stdsc = StandardScaler()
        x_train_std = stdsc.fit_transform(x_train)

        # use random forest
        forest = RandomForestClassifier(n_estimators=500, random_state=42)
        forest.fit(x_train_std, y_train)

        importance = forest.feature_importances_
        indices = np.argsort(importance)[::-1]

        feature_names = self.df_columns[1:]
        for f in range(x_train_std.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feature_names[indices[f]], importance[indices[f]]))

        Plotter.plot_feature_importance(
            x_train_std.shape[1], importance[indices], feature_names[indices],
            FilesystemUtils.get_test_resources_plot_file_name(
                'feature_importance/FeatureImportance.png'
            )
        )
