import os
import shutil
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import pyprind

from definitions import RESOURCES_DATA_DIR


class IMDBDataReader(object):

    def get_data(self):
        # check csv file is existing
        if not os.path.isfile(os.path.join(RESOURCES_DATA_DIR, 'imdb/movie_data.csv')):
            # download dataset from remote
            source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            target = os.path.join(RESOURCES_DATA_DIR, 'imdb/aclImdb_v1.tar.gz')

            if not os.path.isfile(target):
                print('Downloading dataset...')
                import urllib.request
                urllib.request.urlretrieve(source, target, self.report_hook)
                print('\nDone!')
            else:
                print('Dataset has already been downloaded')

            # check if dataset has been uncompressed
            if not os.path.isdir(os.path.join(RESOURCES_DATA_DIR, 'imdb/aclImdb')):
                print('Uncompressing dataset...')
                with tarfile.open(target, 'r:gz') as tar:
                    tar.extractall(os.path.join(RESOURCES_DATA_DIR, 'imdb'))
                print('\nDone!')
            else:
                print('Dataset has been already uncompressed')

            # create a data frame and save to file CSV
            if not os.path.isdir(os.path.join(RESOURCES_DATA_DIR, 'imdb/movie_data.csv')):
                print('Saving dataset as CSV file...')
                base_path = os.path.join(RESOURCES_DATA_DIR, 'imdb/aclImdb')

                labels = {'pos': 1, 'neg': 0}
                progress_bar = pyprind.ProgBar(50000)
                data_frame = pd.DataFrame()
                for s in ('test', 'train'):
                    for l in ('pos', 'neg'):
                        path = os.path.join(base_path, s, l)
                        for file in sorted(os.listdir(path)):
                            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                                txt = infile.read()
                            data_frame = data_frame.append([[txt, labels[l]]], ignore_index=True)
                            progress_bar.update()
                data_frame.columns = ['review', 'sentiment']

                # shuffle data frame
                np.random.seed(42)
                data_frame = data_frame.reindex(np.random.permutation(data_frame.index))
                # save to file
                data_frame.to_csv(os.path.join(RESOURCES_DATA_DIR, 'imdb/movie_data.csv'), index=False,
                                  encoding='utf-8')
                print('\nDone!')
            else:
                print('Dataset already saved as CSV file')

            # delete temporary resources
            print('Removing temporary resources...')
            os.remove(target)
            shutil.rmtree(os.path.join(RESOURCES_DATA_DIR, 'imdb/aclImdb'))
            print('\nDone!')

        return self._load_data()

    def report_hook(self, count, block_size, total_size):
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

        sys.stdout.write(
            '\r%d%% | %d MB | %.2f MB/s | %d sec elapsed' % (percent, progress_size / (1024. ** 2), speed, duration)
        )
        sys.stdout.flush()

    def _load_data(self):
        data_frame = pd.read_csv(os.path.join(RESOURCES_DATA_DIR, 'imdb/movie_data.csv'), encoding='utf-8')
        print('Loaded dataset has shape {}'.format(data_frame.shape))
        print(data_frame.head(10))

        return data_frame


if __name__ == "__main__":
    IMDBDataReader().get_data()
