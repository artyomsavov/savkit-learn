import numpy as np
from collections import Counter


def euclidean_distance(x, y):
    return np.sum((x - y) ** 2)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train] 

        k_nearest_indeces = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indeces] 

        counter = Counter(k_nearest_labels)
        majority_label = counter.most_common(1)[0][0]

        return majority_label
