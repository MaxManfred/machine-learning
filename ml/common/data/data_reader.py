import os
import sys
import time
from abc import abstractmethod

import pandas as pd
from pandas import DataFrame

from definitions import RESOURCES_DATA_DIR


class DataReader(object):

    @abstractmethod
    def get_data(self):
        pass

    def _download_progress_monitor(self, count, block_size, total_size):
        """
        Progress monitor

        :param count:
        :param block_size:
        :param total_size:
        :return:
        """
        global start_time
        if count == 0:
            start_time = time.time()
            return

        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = progress_size / (1024. ** 2 * duration)
        percent = count * block_size * 100. / total_size
        if percent > 100:
            percent = 100

        sys.stdout.write(
            '\r%d%% | %d MB | %.2f MB/s | %d sec elapsed' % (percent, progress_size / (1024. ** 2), speed, duration)
        )
        sys.stdout.flush()

    def _load_data(self, relative_file_path: str = None, use_header: bool = True,
                   displayed_head_size: int = 10) -> DataFrame:
        if use_header:
            data_frame = pd.read_csv(os.path.join(RESOURCES_DATA_DIR, relative_file_path), encoding='utf-8')
        else:
            data_frame = pd.read_csv(os.path.join(RESOURCES_DATA_DIR, relative_file_path), encoding='utf-8',
                                     header=None)

        print('Loaded dataset from CSV has shape {}'.format(data_frame.shape))
        print(data_frame.head(displayed_head_size))

        return data_frame
