import os
from vardialdfs import util
from vardialdfs import config
from sklearn.base import BaseEstimator, TransformerMixin


def lemmatize_sents(X):
    import frog
    frogg = frog.Frog(frog.FrogOptions(morph=False, mwu=False, chunking=False, ner=False))
    new_X = [' '.join([word['lemma'] for word in frogg.process(x)]) for x in X]
    return new_X


def postag_sents(X):
    import frog
    frogg = frog.Frog(frog.FrogOptions(lemma=False, morph=False))
    new_X = [' '.join([word['pos'] for word in frogg.process(x)]) for x in X]
    return new_X


def function_sents(X):
    import frog
    frogg = frog.Frog(frog.FrogOptions(morph=False, mwu=False, chunking=False))
    aux = open('data/ww.txt', 'r').read().splitlines()
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


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.extension = self.get_extension()
        self.train_data, self.test_data = self.load_data()

    def get_extension(self):
        return '.txt'

    def load_data(self):
        train_data, test_data = [], []
        prep_train_path = os.path.splitext(config.TRAIN_FILE)[0] + self.extension
        prep_test_path = os.path.splitext(config.TEST_FILE)[0] + self.extension

        if os.path.isfile(prep_train_path):
            train_data = util.load_data(open(prep_train_path, 'r'))
        if os.path.isfile(prep_test_path):
            test_data = util.load_data(open(prep_test_path, 'r'))
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
        return lemmatize_sents(X)


class POSTagger(Preprocessor):

    def get_extension(self):
        return '.pos'

    def process_data(self, X):
        return postag_sents(X)


class FunctionWords(Preprocessor):

    def get_extension(self):
        return '.fnc'

    def process_data(self, X):
        return function_sents(X)
