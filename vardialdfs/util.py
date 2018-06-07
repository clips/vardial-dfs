from collections import Counter
from sklearn.utils import shuffle


def random_sample_data(X, Y, max_size=None, balance=True):
    """Return a shuffled sample from the dataset.

    Parameters:
    max_size : int
        The maximum length for the returned dataset.
    balance : bool
        Whether the labels in the returned dataset are balanced.

    """
    X, Y = shuffle(X, Y)

    if not max_size or max_size > len(X):
        return X, Y

    if not balance:
        return X[:max_size], Y[:max_size]

    new_X, new_Y = [], []
    counter = Counter()
    labels = set(Y)
    min_labels = min([Y.count(y) for y in labels] + [max_size // len(labels)])
    for i in range(len(X)):
        if counter[Y[i]] == min_labels:
            continue
        new_X.append(X[i])
        new_Y.append(Y[i])
        counter[Y[i]] += 1
    return new_X, new_Y


def load_data(filehandle):
    X, Y = [], []
    for line in filehandle.readlines():
        x, y = line.split('\t')
        X.append(x.strip())
        Y.append(y.strip())
    return X, Y


def save_data(outfile, X, Y):
    for i in range(len(X)):
        outfile.write(X[i] + '\t' + Y[i] + '\n')
    outfile.close()


def save_labels(outfile, Y):
    for y in Y:
        outfile.write(y + '\n')
    outfile.close()
