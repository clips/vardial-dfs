from sklearn.utils import shuffle


def random_sample_data(X, Y, max_size=None, balance=True):
    X, Y = shuffle(X, Y)
    if not max_size or max_size > len(X):
        return X, Y
    if not balance:
        return X[:max_size], Y[:max_size]
    new_X, new_Y = [], []
    number_of_labels = len(set(Y))
    min_labels = max_size // number_of_labels
    for i in range(len(X)):
        if new_Y.count(Y[i]) == min_labels:
            continue
        new_X.append(X[i])
        new_Y.append(Y[i])
    return new_X, new_Y


def load_data(filehandle):
    X, Y = [], []
    for line in filehandle.readlines():
        x, y = line.split('\t')
        X.append(x.strip())
        Y.append(y.strip())
    return X, Y
