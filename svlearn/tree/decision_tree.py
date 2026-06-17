from collections import Counter
from typing import Self
import numpy as np

from ..base import BaseEstimator, Features, Target, Prediction


class Node:
    def __init__(
        self, 
        feature: int | None = None, 
        threshold: float | None = None, 
        left: "Node | None" = None, 
        right: "Node | None" = None, 
        *, 
        value: int | float | None = None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        return self.value is not None 


class DecisionTree(BaseEstimator):
    def __init__(
        self, 
        min_samples_split: int = 2, 
        min_impurity_decrease: float = 0.0, 
        max_depth: int = 100, 
        n_features: int | None = None
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = n_features
        self.root: Node | None = None

    def fit(self, X: Features, y: Target) -> Self:
        if not self.n_features:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(X.shape[1], self.n_features) 

        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X: Features, y: Target, depth: int = 0) -> Node:
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

    def _best_split(self, X: Features, y: Target, 
                    feature_idxs: np.ndarray) -> tuple[int | None, float | None, float]:
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold, best_gain

    def _information_gain(self, X_column: np.ndarray, y: Target, 
                          threshold: float) -> float:
        # parent entropy
        parent_entropy = self._entropy(y)

        # child avg. weighted entropy
        left_idxs, right_idxs = self._split(X_column, threshold)

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs) 
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        children_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - children_entropy

    def _split(self, X_column: np.ndarray, 
                threshold: float) -> tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y: Target) -> float:
        counts = np.bincount(y)
        p = counts[counts > 0] / len(y)
        return - (p @ np.log(p)) 
        
    def _most_common_label(self, y: Target) -> int | float:
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X: Features) -> Prediction:
        if self.root is None: 
            raise RuntimeError('Before calling predict, you must fit the model.')

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int | float:
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

