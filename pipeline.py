import config
from sklearn.svm import LinearSVC
from features import SentenceLengthFeature
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filehandle):
    X, Y = [], []
    for line in filehandle.readlines():
        x, y = line.split('\t')
        X.append(x.strip())
        Y.append(y.strip())
    return X, Y


def pipeline_config(clf):
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('vec', TfidfVectorizer(ngram_range=(1, 3))),
            ('lexicon', SentenceLengthFeature()),
        ])),
        ('clf', clf)])

    return pipeline


def run():
    train_handle = open(config.TRAIN_FILE, 'r')
    dev_handle = open(config.TEST_FILE, 'r')
    X, Y = load_data(train_handle)
    X_dev, Y_dev = load_data(dev_handle)

    clf = LinearSVC()
    pipeline = pipeline_config(clf)

    pipeline.fit(X, Y)
    Y_pred = pipeline.predict(X_dev)
    score = accuracy_score(Y_dev, Y_pred)

    print("Accuracy: {}".format(score))


run()