import numpy as np

from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_split_size=2, max_depth=100, n_features=None): 
       self.min_split_size = min_split_size
       self.max_depth = max_depth
       self.n_features = n_features
       self.root = None

    def fit(self, X, y):
        if self.n_features is None:
            self.n_features = X.shape[1] 
        else:
            min(X.shape[1], self.n_features)

        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape 

        # stopping criteria
        if (depth >= self.max_depth
            or X.shape[0] < self.min_split_size
        ):
            counter = Counter(y)
            value = counter(1)[0][0]
            return Node(value=value) 

        # subset random
        feature_idxs = np.random.choice(n_features, n_features, replace=False)

        # find best splits
        best_split, best_threshold = self._best_split(feature_idxs, X)
        
        # split
        left_idxs, right_idxs = self._split(X[best_split], best_threshold) 
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_split, best_threshold, left, right) 
        

    def _best_split(self, feature_idxs, X):
        best_gain = -1
        best_split, best_threshold = None, None 

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            for threshold in X_column:
                curr_information_gain = self._information_gain(X_column, y, threshold) 
                if curr_information_gain > best_gain:
                    best_split = feature_idx
                    best_threshold = threshold
                    best_gain = curr_information_gain 

        return best_split, best_threshold

    def _information_gain(self, X_column, y, threshold):
        left_idx, right_idx = self._split(X_column, threshold)
        # parent
        e_p = self._entropy(y)

        # chilkd
        e_l = self._entropy(y[left_idx])
        e_r = self._entropy(y[right_idx])
        n_l = len(left_idx) 
        n_r = len(right_idx)
        n = len(X_column)

        information_gain = e_p - (n_l / n) * e_l + (n_r / n) * e_r
        return information_gain

    def _split(self, X_column, threshold):
        """pay attention"""
        left_idx = np.argwhere(X_column < threshold)
        right_idx = np.argwhere(X_column >= threshold)
        return left_idx, right_idx

    def _entropy(self, y):
        counts = np.bincount(y)
        p = counts[counts > 0] / len(y)
        return - (p @ np.log(p))

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter(1)[0][0] 

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(self, x, node.left)
        return self._traverse_tree(self, x, node.right)
        

