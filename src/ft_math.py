import numpy as np


def mean(X):
    """ Average value of a series of data """
    X = np.array(X)
    return (np.sum(X) / len(X))


def softmax(X):
    """ Exponential normalized function """
    return (np.exp(X) / np.sum(np.exp(X)))


def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def sigmoid(X):
    """ Sigmoid Function """
    return (1 / (1 + np.exp(-X)))


def relu(X):
    if (isinstance(X, np.ndarray)):
        return np.array([max(n, 0) for n in X])
    else:
        return max(X, 0)


def cross_entropy(y, p):
    eps = 1e-15
    c = -sum((y * np.log(p + eps)) + ((1 - y) * np.log(1 - p + eps))) / len(y)
    return c


def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))
    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5])
    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    print(sigmoid(X))
    print(softmax(z))
    y = np.array([0.1, 0.3])
    ypred = np.array([0.5, 0.6])

    print()
    print(cross_entropy(y, ypred))
    x2 = np.array([-1, -2.8, -0, 1.4, 1])
    print(relu(x2))
