import numpy as np
from collections import Counter
from svlearn import DecisionTree


class RandomForest:

    def __init__(self, n_trees=30, max_depth=100, min_samples_split=2, 
                 min_impurity_decrease=0.0, n_features=None):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = n_features
        self.trees = None 

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees): 
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_impurity_decrease = self.min_impurity_decrease,
                n_features = self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        sample_idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, y_sample = X[sample_idxs], y[sample_idxs] 
        return X_sample, y_sample

    def predict(self, X):
        y_pred_unmcld = np.array([tree.predict(X) for tree in self.trees]).T
        y_pred = np.array(
                [self._most_common_label(predictions) 
                 for predictions in y_pred_unmcld]
        )
        return y_pred

    def _most_common_label(self, predictions):
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]

