from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class SentenceLengthFeature(BaseEstimator, TransformerMixin):
    """outputs sentence length per document"""

    def transform(self, X, y=None):
        return [[len(x.split())] for x in X]

    def fit(self, X, y=None):
        return self


class POSVectorizer(CountVectorizer):
    """ adds postags, learns weights """

    def postag(self, X):
        import frog
        frogg = frog.Frog(frog.FrogOptions(lemma=False, morph=False))
        new_X = [' '.join([word['pos'] for word in frogg.process(x)]) for x in X]
        return new_X

    def transform(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer, self).transform(X)

    def fit(self, X, y=None):
        X = self.postag(X)
        return super(POSVectorizer, self).fit(X, y)
