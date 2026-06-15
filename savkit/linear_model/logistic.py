import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = X @ self.weights + self.bias
            y_pred = sigmoid(linear_pred)

            dw = X.T @ (y_pred - y) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_pred = X @ self.weights + self.bias
        y_pred = sigmoid(linear_pred)
        labels = [1 if y > 0.5 else 0 for y in y_pred]
        return labels

