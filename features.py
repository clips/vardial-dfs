from sklearn.base import BaseEstimator, TransformerMixin


class SentenceLengthFeature(BaseEstimator, TransformerMixin):
    """outputs sentence length per document"""

    def transform(self, X, y=None):
        return [[len(x.split())] for x in X]

    def fit(self, X, y=None):
        return self
