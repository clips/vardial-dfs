import os
import frog
from vardialdfs import util
from vardialdfs import config
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):

    """This is a parent class for preprocessing data in the pipeline."""

    def __init__(self):
        self.extension = self.get_extension()
        self.train_data, self.test_data = self.load_data()

    def get_extension(self):
        return '.txt'

    def get_train_path(self):
        train_path = os.path.splitext(config.TRAIN_FILE)[0] + self.extension
        return train_path

    def get_test_path(self):
        test_path = os.path.splitext(config.TEST_FILE)[0] + self.extension
        return test_path

    def load_data(self):
        """Load preprocessed data if available, otherwise do preprocessing."""
        train_data, test_data = [], []
        train_path = self.get_train_path()
        test_path = self.get_test_path()

        if os.path.isfile(train_path):
            train_data, _ = util.load_data(open(train_path, 'r'))
        if os.path.isfile(test_path):
            test_data, _ = util.load_data(open(test_path, 'r'))
        return train_data, test_data

    def process_data(self, X):
        return X

    def transform(self, X, y=None):
        if len(X) == len(self.train_data):
            return self.train_data
        if len(X) == len(self.test_data):
            return self.test_data
        return self.process_data(X)

    def fit(self, X, y=None):
        return self


class Lemmas(Preprocessor):

    def get_extension(self):
        return '.lem'

    def process_data(self, X):
        frogg = frog.Frog(frog.FrogOptions(morph=False, mwu=False, chunking=False, ner=False))
        new_X = [' '.join([word['lemma'] for word in frogg.process(x)]) for x in X]
        return new_X


class POSTagger(Preprocessor):

    def get_extension(self):
        return '.pos'

    def process_data(self, X):
        import frog
        frogg = frog.Frog(frog.FrogOptions(lemma=False, morph=False))
        new_X = [' '.join([word['pos'] for word in frogg.process(x)]) for x in X]
        return new_X


class FunctionWords(Preprocessor):

    def get_extension(self):
        return '.fnc'

    def process_data(self, X):
        """Filter data. Leave only articles, pronouns, conjunctions and auxiliary verbs."""
        frogg = frog.Frog(frog.FrogOptions(morph=False, mwu=False, chunking=False))
        aux = open(config.VERB_FILE, 'r').read().splitlines()
        new_X = []
        for x in X:
            new_x = []
            output = frogg.process(x)
            for word in output:
                if word['pos'][:3] not in ['LID', 'VNW', 'VG(', 'WW(']:
                    continue
                if word['pos'][:2] == 'WW':
                    if word['lemma'] in aux:
                        new_x.append(word['lemma'])
                    continue
                new_x.append(word['text'].lower())
            new_X.append(new_x)
        return new_X
