import numpy as np
from collections import Counter


class Node:
    
    def __init__(self, feature=None, 
                 threshold=None, left=None, right=None, *, value=None):

       self.feature = feature
       self.threshold = threshold
       self.left = left
       self.right = right
       self.value = value

    def is_leaf(self):
        return self.value is not None 

class DecisionTree:

    def __init__(self, min_samples_split=2, 
                 min_impurity_decrease=0.0, max_depth=100, n_features=None):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = n_features
        self.root = None


    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) 
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        labels = np.unique(y)

        # criteria
        if (depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(labels) == 1
        ):
            leaf_label = self._most_common_label(y)
            return Node(value=leaf_label)

        # random subset
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        
        # find splits and check criteria impurity 
        split_idx, split_threshold, split_gain = self._best_split(X, y, feature_idxs)
        
        if split_gain < self.min_impurity_decrease:
            leaf_label = self._most_common_label(y)
            return Node(value=leaf_label)

        # create children
        left_idxs, right_idxs = self._split(X[:, split_idx], split_threshold)

        left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(split_idx, split_threshold, left_child, right_child)

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            for threshold in X_column:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold, gain

    def _information_gain(self, X_column, y, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # child avg. weighted entropy
        left_idxs, right_idxs = self._split(X_column, threshold)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs) 
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        children_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - children_entropy

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        counts = np.bincount(y)
        p = counts[counts > 0] / len(y)
        return - (p @ np.log(p)) 
        
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
