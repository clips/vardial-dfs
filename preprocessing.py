import os
import util
import config
from sklearn.base import BaseEstimator, TransformerMixin


def function_sents(X):
    import frog
    frogg = frog.Frog(frog.FrogOptions(morph=False, mwu=False, chunking=False))
    aux = open('data/ww.txt', 'r').read().splitlines()
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
        print(' '.join(new_x), flush=True)


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


class POSTagger(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.train_data, self.test_data = self.load_pos()

    def load_pos(self):
        train_data, test_data = [], []
        prep_train_path = os.path.splitext(config.TRAIN_FILE)[0] + '.pos'
        prep_test_path = os.path.splitext(config.DEV_FILE)[0] + '.pos'

        if os.path.isfile(prep_train_path):
            train_data = util.load_data(open(prep_train_path, 'r'))
        if os.path.isfile(prep_test_path):
            test_data = util.load_data(open(prep_test_path, 'r'))
        return train_data, test_data

    def transform(self, X, y=None):
        if len(X) == len(self.train_data):
            return self.train_data
        if len(X) == len(self.test_data):
            return self.test_data
        return postag_sents(X)

    def fit(self, x, y=None):
        return self


class FunctionWords(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.train_data, self.test_data = self.load_functionwords()

    def load_functionwords(self):
        train_data, test_data = [], []
        prep_train_path = os.path.splitext(config.TRAIN_FILE)[0] + '.fnc'
        prep_test_path = os.path.splitext(config.DEV_FILE)[0] + '.fnc'

        if os.path.isfile(prep_train_path):
            train_data = util.load_data(open(prep_train_path, 'r'))
        if os.path.isfile(prep_test_path):
            test_data = util.load_data(open(prep_test_path, 'r'))
        return train_data, test_data

    def transform(self, X, y=None):
        if len(X) == len(self.train_data):
            return self.train_data
        if len(X) == len(self.test_data):
            return self.test_data
        return function_sents(X)

    def fit(self, X, y=None):
        return self
