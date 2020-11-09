import numpy as np

def softmax(X):
    """ Exponential normalized function """
    return (np.exp(X) / np.sum(np.exp(X)))

def sigmoid(X):
    """ Sigmoid Function """
    return (1 / 1 + np.exp(-X))


if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5])
    z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
    print(sigmoid(X))
    print(softmax(z))