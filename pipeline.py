import util
import config
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from preprocessing import POSTagger,Lemmas
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer


def base_classifier(clf, features):
    pipeline = Pipeline([
        ('features', features),
        ('clf', clf)])

    return pipeline


def run():
    train_handle = open(config.TRAIN_FILE, 'r')
    dev_handle = open(config.DEV_FILE, 'r')
    X, Y = util.load_data_and_labels(train_handle)
    X_dev, Y_dev = util.load_data_and_labels(dev_handle)

    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm)

    feature_list = [
        TfidfVectorizer(ngram_range=(1, 3)),
        Pipeline([('postag', POSTagger()),
                  ('vectorizer', TfidfVectorizer(ngram_range=(3, 6), tokenizer=lambda x: x.split()))]),
        Pipeline([('postag', Lemmas()),
                  ('vectorizer', TfidfVectorizer(ngram_range=(1, 4)))])
    ]

    Y_proba = np.zeros((len(X_dev), 2))

    for features in feature_list:
        pipeline = base_classifier(clf, features)
        pipeline.fit(X, Y)
        Y_proba = np.sum([Y_proba, pipeline.predict_proba(X_dev)], axis=0)

    labels = clf.classes_
    Y_pred = [labels[0] if proba[0] > proba[1] else labels[1] for proba in Y_proba]
    score = accuracy_score(Y_dev, Y_pred)

    print("Accuracy: {}".format(score))


run()
