import numpy as np


def LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0 

        for _ in range(n_iters):
            y_pred = X @ self.weights + self.bias

            dw = X.T @ (y_pred - y) / N
            db = np.sum(y_pred - y) / N 

            self.weights = self.weights - lr * dw
            self.bias = self.bias - lr * db

    def predict(self, X):
        y_pred = X @ self.weights + self.bias
        return y_pred 

