from ft_math import sigmoid, relu
import numpy as np
import sklearn.preprocessing


class layer:

    def __init__(self, w, b, a=sigmoid):
        self.w = np.array(w)
        self.b = np.array(b)
        self.a = a

    def out(self, X):
        return self.a(np.sum(self.w * X) + self.b)


if __name__ == "__main__":
    w = [0, 0, 1, 0, 0]
    b = 0.0
    l0 = layer(w, b)
    X = [0.1, 0.2, 0.3, -0.4, 0.1]
    print(l0.out(X))
