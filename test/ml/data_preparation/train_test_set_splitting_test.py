from sklearn.model_selection import train_test_split

from test.ml.common.wine_dataset_test import WineDatasetTest


class TrainTestSplittingTest(WineDatasetTest):

    @classmethod
    def setUpClass(cls):
        cls.switcher = {
            'test_train_test_set_splitting_with_scikit': cls.load_wine_data_set
        }

    def test_train_test_set_splitting_with_scikit(self):
        # then create train and test sets
        # By setting test_size = 0.3, we assign 30 percent of the wine samples to X_test and Y_test, and the remaining
        # 70 percent of the samples are assigned to X_train and Y_train, respectively. Providing the class label array y
        # as an argument to stratify ensures that both training and test datasets have the same class proportions as the
        # original dataset.
        X_train, X_test, Y_train, Y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=0, stratify=self.y)

        print("\n")
        print('Training set shape (70% of total): ', X_train.shape)
        print('Training labels size (70% of total): ', Y_train.size)

        print('Test set shape (30% of total): ', X_test.shape)
        print('Test labels size (30% of total): ', Y_test.size)
