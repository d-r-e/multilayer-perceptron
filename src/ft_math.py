import numpy as np
from math import log2

def mean(X):
    """ Average value of a series of data """
    X = np.array(X)
    return (np.sum(X) / len(X))


def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def cross_entropy(y_hat, y):
    eps = 1e-15
    y = np.squeeze(y)
    c = sum((y * np.log(y_hat + eps)) + ((1 - y) * np.log(1 - y_hat + eps))) / -len(y)
    return c

def sigmoid(X):
    """ Sigmoid Function """
    return (1 / (1 + np.exp(-X)))


def relu(X):
    if (isinstance(X, np.ndarray)):
        return np.array([max(n, 0) for n in X])
    else:
        return max(X, 0)


def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits) + 1e-14, axis=-1))
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
