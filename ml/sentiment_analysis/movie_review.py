import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from ml.common.data.imdb_data_reader import IMDBDataReader


class MovieReview(object):

    def __init__(self):
        nltk.download('stopwords')
        self.imdb_data_reader = IMDBDataReader()

    def remove_html(self, text: str):
        # Via the first regex <[^>]*> in the preceding code section, we tried to remove all of the HTML markup from the
        # movie reviews.
        # After we removed the HTML markup, we used a slightly more complex regex to find emoticons, which we t
        # emporarily stored as emoticons.
        # Next, we removed all non-word characters from the text via the regex [\W]+ and converted the text into
        # lowercase characters: in the context of this analysis, it is a simplifying assumption that the letter case
        # does not contain information that is relevant for sentiment analysis.

        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

        return text

    def tokenize_and_stem(self, text: str):
        porter = PorterStemmer()
        return [porter.stem(word) for word in text.split()]

    def remove_stopwords(self, text: str):
        stop = stopwords.words('english')
        return [w for w in self.tokenize_and_stem(text) if w not in stop]

    def train_logistic_regression(self):
        # load data
        data_frame = self.imdb_data_reader.get_data()

        # identity train and test sets
        x_train = data_frame.loc[:25000, 'review'].values
        y_train = data_frame.loc[:25000, 'sentiment'].values

        x_test = data_frame.loc[25000:, 'review'].values
        y_test = data_frame.loc[25000:, 'sentiment'].values

        # Next, we will use a GridSearchCV object to find the optimal set of parameters for our logistic regression model
        # using 5-fold stratified cross-validation:

        # TODO: COMPLETE!