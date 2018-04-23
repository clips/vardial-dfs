import os
import util
import config
from sklearn.base import BaseEstimator, TransformerMixin


class POSTagger(BaseEstimator, TransformerMixin):

    def load_saved_pos(self):
        train_data, test_data = [], []
        prep_train_path = os.path.splitext(config.TRAIN_FILE)[0] + '.pos'
        prep_test_path = os.path.splitext(config.TEST_FILE)[0] + '.pos'

        if os.path.isfile(prep_train_path):
            train_data, _ = util.load_data(open(prep_train_path, 'r'))
        if os.path.isfile(prep_test_path):
            test_data, _ = util.load_data(open(prep_test_path, 'r'))

        return train_data, test_data

    def tag_sents(self, X):
        import frog
        frogg = frog.Frog(frog.FrogOptions(lemma=False, morph=False))
        new_X = [' '.join([word['pos'] for word in frogg.process(x)]) for x in X]
        return new_X

    def transform(self, X, y=None):
        self.train_data, self.test_data = self.load_saved_pos()
        if len(X) == len(self.train_data):
            return self.train_data
        if len(X) == len(self.test_data):
            return self.test_data
        return self.tag_sents(X)

    def fit(self, x, y=None):
        return self
