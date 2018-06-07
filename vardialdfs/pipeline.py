import numpy as np
from vardialdfs import util
from vardialdfs import config
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from vardialdfs.preprocessing import POSTagger
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def base_classifier(clf, features):
    classifier = Pipeline([
        ('features', FeatureUnion(features)),
        ('clf', clf)])
    return classifier


def run_pipeline(meta_method):
    train_handle = open(config.TRAIN_FILE, 'r')
    test_handle = open(config.TEST_FILE, 'r')
    X_train, Y_train = util.load_data_and_labels(train_handle)
    X_test, Y_test = util.load_data_and_labels(test_handle)

    feature_list = [
        ('word_ngrams', TfidfVectorizer(ngram_range=(1, 3))),
        ('pos_ngrams', Pipeline([('postag', POSTagger()),
                  ('vectorizer', TfidfVectorizer(ngram_range=(3, 6), tokenizer=lambda x: x.split()))])),
    ]

    if meta_method == 'meta_classifier':

        svm = LinearSVC()
        calibrated_cv = CalibratedClassifierCV(svm)

        meta_train = np.array([]).reshape(len(X_train), 0)
        meta_test = np.array([]).reshape(len(X_test), 0)

        for features in feature_list:
            classifier = base_classifier(calibrated_cv, [features])
            classifier.fit(X_train, Y_train)
            meta_train = np.column_stack([meta_train, classifier.predict_proba(X_train)[:, 0]])
            meta_test = np.column_stack([meta_test, classifier.predict_proba(X_test)[:, 0]])

        meta_classifier = LinearDiscriminantAnalysis()
        meta_classifier.fit(meta_train, Y_train)
        Y_pred = meta_classifier.predict(meta_test)

    elif meta_method == 'meta_vote':

        svm = LinearSVC()
        calibrated_cv = CalibratedClassifierCV(svm)

        Y_proba = np.zeros((len(X_test), 2))

        for features in feature_list:
            pipeline = base_classifier(calibrated_cv, [features])
            pipeline.fit(X_train, Y_train)
            Y_proba = np.sum([Y_proba, pipeline.predict_proba(X_test)], axis=0)

        labels = calibrated_cv.classes_
        Y_pred = [labels[0] if p[0] > p[1] else labels[1] for p in Y_proba]

    else:
        svm = LinearSVC()
        pipeline = base_classifier(svm, feature_list)
        pipeline.fit(X_train, Y_train)
        Y_pred = pipeline.predict(X_test)

    f_macro = f1_score(Y_test, Y_pred, average='macro')
    print(f_macro)
