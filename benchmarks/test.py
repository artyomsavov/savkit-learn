from svlearn import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

bc = datasets.load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, random_state=1234
)

clf = DecisionTree(min_impurity_decrease=0.001)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(np.mean(y_test == y_pred))
