import numpy as np


class Perceptron:
    def __init__(self):
        self.X = None
        self.y = None
        self.w = None

    def fit(self, X, y):
        self.X = np.c_[X, np.ones(X.shape[0])]
        self.y = y
        self.w = np.zeros(self.X.shape[1])
        while True:
            temp = self.w @ self.X.T * self.y
            choose = np.argwhere(temp <= 0)
            if choose.size == 0:
                return
            else:
                i = choose[0]
                self.w = self.w + self.y[i] @ self.X[i]

    def predict(self, x):
        if np.dot(x, self.w[:-1])+self.w[self.w.size-1] > 0:
            return 1
        else:
            return -1
