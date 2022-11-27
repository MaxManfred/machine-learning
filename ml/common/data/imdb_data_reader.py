import os
import shutil
import tarfile

import numpy as np
import pandas as pd
import pyprind

from definitions import RESOURCES_DATA_DIR
from ml.common.data.data_reader import DataReader


class IMDBDataReader(DataReader):

    def get_data(self):
        # check csv file is existing
        if not os.path.isfile(os.path.join(RESOURCES_DATA_DIR, 'imdb/movie_data.csv')):
            # download dataset from remote
            source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
            target = os.path.join(RESOURCES_DATA_DIR, 'imdb/aclImdb_v1.tar.gz')

            if not os.path.isfile(target):
                print('Downloading dataset...')
                import urllib.request
                urllib.request.urlretrieve(source, target, self._download_progress_monitor)
                print('\nDone!')
            else:
                print('Dataset has already been downloaded')

            # check if dataset has been uncompressed
            if not os.path.isdir(os.path.join(RESOURCES_DATA_DIR, 'imdb/aclImdb')):
                print('Uncompressing dataset...')
                with tarfile.open(target, 'r:gz') as tar:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(tar, os.path.join(RESOURCES_DATA_DIR,"imdb"))
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

        return self._load_data(relative_file_path='imdb/movie_data.csv')


if __name__ == "__main__":
    IMDBDataReader().get_data()
