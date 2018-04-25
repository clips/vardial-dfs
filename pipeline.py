import util
import config
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from preprocessing import POSTagger, Lemmas, FunctionWords
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def base_classifier(features):
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm)
    classifier = Pipeline([
        ('features', features),
        ('clf', clf)])
    return classifier


def run():
    train_handle = open(config.TRAIN_FILE, 'r')
    dev_handle = open(config.DEV_FILE, 'r')
    X, Y = util.load_data_and_labels(train_handle)
    X_dev, Y_dev = util.load_data_and_labels(dev_handle)

    feature_list = [
        TfidfVectorizer(ngram_range=(1, 3)),
        Pipeline([('postag', POSTagger()),
                  ('vectorizer', TfidfVectorizer(ngram_range=(3, 6), tokenizer=lambda x: x.split()))]),
        Pipeline([('lemmas', Lemmas()),
                  ('vectorizer', TfidfVectorizer(ngram_range=(1, 2)))]),
        Pipeline([('functionwords', FunctionWords()),
                  ('vectorizer', TfidfVectorizer(ngram_range=(1, 2)))])
    ]

    meta_train = np.array([]).reshape(len(X), 0)
    meta_dev = np.array([]).reshape(len(X_dev), 0)

    for features in feature_list:
        classifier = base_classifier(features)
        classifier.fit(X, Y)
        meta_train = np.column_stack([meta_train, classifier.predict_proba(X)[:,0]])
        meta_dev = np.column_stack([meta_dev, classifier.predict_proba(X_dev)[:,0]])

    meta_classifier = LinearDiscriminantAnalysis()
    meta_classifier.fit(meta_train, Y)
    Y_pred = meta_classifier.predict(meta_dev)
    score = accuracy_score(Y_dev, Y_pred)

    print("Accuracy: {}".format(score))


run()
