import util
import config
from sklearn.svm import LinearSVC
from preprocessing import POSTagger
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer


def pipeline_config(clf):
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('word_ngrams', TfidfVectorizer(ngram_range=(1, 3))),
            ('pos_ngrams', Pipeline([
                ('postag', POSTagger()),
                ('vectorizer', TfidfVectorizer(ngram_range=(3, 3), tokenizer=lambda x: x.split()))
            ])),
        ])),
        ('clf', clf)])

    return pipeline


def run():
    train_handle = open(config.TRAIN_FILE, 'r')
    test_handle = open(config.TEST_FILE, 'r')
    X, Y = util.load_data(train_handle)
    X_test, Y_test = util.load_data(test_handle)

    clf = LinearSVC()
    pipeline = pipeline_config(clf)

    pipeline.fit(X, Y)
    Y_pred = pipeline.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)

    print("Accuracy: {}".format(score))


run()
