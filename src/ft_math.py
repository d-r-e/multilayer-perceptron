import numpy as np


def mean(X):
    """ Average value of a series of data """
    X = np.array(X)
    return (np.sum(X) / len(X))


def softmax(X):
    """ Exponential normalized function """
    return (np.exp(X) / np.sum(np.exp(X)))


def sigmoid(X):
    """ Sigmoid Function """
    return (1 / (1 + np.exp(-X)))


def relu(X):
    return np.max(X, 0)


def cross_entropy(y, p):
    eps = 1e-15
    c = -sum((y * np.log(p + eps)) + ((1 - y) * np.log(1 - p + eps))) / len(y)
    return c


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5])
    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    print(sigmoid(X))
    print(softmax(z))
    y = np.array([0.1, 0.3])
    ypred = np.array([0.5, 0.6])

    print()
    print(cross_entropy(y, ypred))
