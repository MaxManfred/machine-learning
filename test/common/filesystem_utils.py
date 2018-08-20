import os

from definitions import TEST_RESOURCES_PLOT_DIR, RESOURCES_DATA_DIR


class FilesystemUtils(object):

    @staticmethod
    def get_resources_data_file_name(file_name: str):
        return os.path.join(RESOURCES_DATA_DIR, file_name)

    @staticmethod
    def get_test_resources_data_file_name(file_name: str):
        return os.path.join(TEST_RESOURCES_PLOT_DIR, file_name)

    @staticmethod
    def get_test_resources_plot_file_name(file_name: str):
        return os.path.join(TEST_RESOURCES_PLOT_DIR, file_name)
