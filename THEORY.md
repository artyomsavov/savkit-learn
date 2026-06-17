# K Nearest Neighbours

### Steps
**Given a data point:**
- Calculate its distance from all other data points int the dataset
- Get the closest K points
- _Regression:_ Get the average of their values
- _Classification:_ Get the label with majority vote

# Linear Regression

### Estimation

$$
\hat{y} = X \omega + b
$$
### Calculating Error
$$
MSE = J(\omega, b) = \frac{1}{N} \sum_{i=1}^n (y_i - (\omega x_i + b))^2
$$
### Updating parameters
$$
J'(\omega, b) = 
\begin{bmatrix}
\frac{\partial J}{\partial \omega} \\[6px]
\frac{\partial J}{\partial b}
\end{bmatrix}
=
\begin{bmatrix}
\frac{1}{N} \sum - 2 x_i (y_i - (\omega x_i + b)) \\[6px]
\frac{1}{N} \sum - 2 (y_i - (\omega x_i + b))
\end{bmatrix}
=
\begin{bmatrix}
\frac{1}{N} \sum 2 x_i (\hat{y_i} - y_i) \\[6px]
\frac{1}{N} \sum 2 (\hat{y_i} - y_i)
\end{bmatrix}
=
\begin{bmatrix}
\frac{2}{N} X^T (\hat{y} - y)  \\[6px]
\frac{2}{N} \sum (\hat{y_i} - y_i)
\end{bmatrix}
= 
\begin{bmatrix}
dw \\[6px]
db
\end{bmatrix}
$$
### Gradient Descent
$$
\omega_{n+1} = \omega_n - \alpha \cdot dw
$$
$$
b_{n+1} = b_n - \alpha \cdot db
$$
### Steps
##### **Training:**
- Initialize weight as zero
- Initialize bias as zero
**Given a data point**:
- Predict result by using $\hat{y} = \omega x + b$
- Calculate error
- Use gradient descent to figure out new weight and bias values
- Repeat n times
##### **Testing:**
Given a data point:
- Put in the values from the data point into the equation $\hat{y} = \omega x + b$
# Logistic Regression
### Estimation
$$
\hat{y} = h_{\theta}(X) = \sigma(\hat{y}_{linear}) = \frac{1}{1 + e^{\hat{y}_{linear}}} = \frac{1}{1 + e^{-wX - b}}
$$
### Calculating error
$$
CrossEntropy = J(\omega, b) = J(\theta) = - \frac{1}{N} \sum_{i=1}^N [y_i log(h_{\theta}(x_i)) + (1-y_i) log(1 - h_{\theta}(x_i))]
$$
### Updating parameters
$$
J'(\theta) = 
\begin{bmatrix}
\frac{\partial J}{\partial \omega} \\[6px] 
\frac{\partial J}{\partial b}
\end{bmatrix} = 
\begin{bmatrix}
\frac{1}{N} X^T (\hat{y} - y) \\[6px] 
\frac{1}{N} \sum (\hat{y}_i - y_i)
\end{bmatrix}
$$
### Gradient Descent
$$
\omega_{n+1} = \omega_n - \alpha \cdot dw
$$
$$
b_{n+1} = b_n - \alpha \cdot db
$$
### Steps
##### **Training:**
- Initialize weight as zero
- Initialize bias as zero
**Given a data point**:
- Predict result by using $\hat{y} = \frac{1}{1 + e^{- \omega x - b}}$
- Calculate error
- Use gradient descent to figure out new weight and bias values
- Repeat n times
##### **Testing:**
Given a data point:
- Put in the values from the data point into the equation $\hat{y} = \frac{1}{1 + e^{- \omega x - b}}$
- Choose the label based on the higher probability

# Decision Trees
### What needs to be decided on?
- Split feature
- Split point
- When to stop splitting
### Steps
##### **Training:**
Given the whole dataset:
- Calculate **information gain** with each possible split
- Divide set with that feature and value that gives the most IG
- Divide tree and do the same for all created branches ...
- ... until a **stopping criteria** is reached
##### **Testing:**
Given a data point:
- Follow the tree until you reach a leaf node
- Return the most common class label

### Terms
- Information gain
$$
IG = E(parent) - [weighted\space average] \cdot E(children)
$$
- Entropy
$$
E = - \sum p(X) \cdot log_2 (p(X)), \quad p(X) = \frac{\#x}{n}
$$
- Stopping criteria: 
	- maximum depth
	- minimum number of samples a node can have
	- min impurity decrease — min entropy changes

# Random forest

### Steps
##### **Training:**
Given the whole dataset:
- Get a subset of the dataset
- Create a decision tree
- Repeat for as many times as the number of trees
##### **Testing:**
Given a data point:
- Get the predictions from each tree
- _Classification:_ hold a majority vote
- _Regression:_ get the mean of the predictions

# Naive Bayes

	The Naive Bayes claffifier is a "probabilistic classifier" based on applying Bayes' threorem with strong (naive) independence assumptions between the features.

$$
P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
$$
**Assume that features are mutually conditional independent**
$$
P(y|X) = \frac{P(x_1|y) \cdot P(x_2|y) \cdot \ldots \cdot P(x_n|y) \cdot P(y)}{P(X)} \propto P(x_1|y) \cdot P(x_2|y) \cdot \ldots \cdot P(x_n|y)
$$
**Select class with highest posterior probability**
$$
y = argmax_yP(y|X) = P(x_1|y) \cdot P(x_2|y) \cdot \ldots \cdot P(x_n|y)
$$
$$
y = argmax_y \left [log(P(y))+\sum_i^n log(P(x_i|y)) \right]
$$
- $P(y)$ — Prior probability ~ Frequency of each class
- $P(x_i|y)$ — Class conditional probability ~ Model with Gaussian distribution
$$
P(x_i|y) = \frac{1}{\sqrt{2 \pi \sigma_y^2}} exp \left (- \frac{(x_i - \mu_y)^2}{2\sigma_y^2} \right)
$$
### Steps
##### Training:
- Calculate mean, variation and prior (frequency) for each class
##### Predictions:
- Calculate posterior for each class
- Choose class with highest posterior probability

# Principal Component Analysis

	PCA is unsupervised learning method that is ofthen used to reduce the dimensionality of the dataset by tranforming a large set into a lower dimensional set that still contains most of the information of the large set.

	PCA finds a new set of dimensions such that all the dimensions are orthogonal (and hence linearly independent) and ranked according to the variance of data along them

	Find a transofrmation such that
- The transformed features are **linearly independent**
- **Dimensionality ca be reduced** by taking only the dimensions with the highest importance
- Those newly found dimensions should **minimize the projection error**
- The projected point should have maximum spread, i.e. **maximum variance**

$$
Var(X) = \frac{1}{n} \sum (X_i - \bar{X})^2
$$
$$
Cov(X, Y) = \frac{1}{n} \sum (X_i - \bar{X})(Y_i - \bar{Y})^T
$$
$$
Av= \lambda v
$$
### Steps
##### Training:
- Calculate $Cov(X, X)$
- Calculate eigenvectors and eigenvalues of the covariance matrix
- Sort the eigenvectors according to their eigenvalues in decreasing order
- Choose first k eigenvectors and that will be the new k dimensions
##### Transform:
- Transform the original n-dimensional data point into k dimensions (projections with dot product)

### Perceptron

	The Perceptron is an algorighm for supervised learning of binary classifiers. It can be seen as a single unit of an aritficial neural network and is also know as the Prototype for Neural Nets.
	Single Layer Perceptron: Can learn only linearly separable patterns.
	Multilayer Perceptron: Can learn more complex patterns.

**Linear model**
$$
f(x) = \omega^T x + b
$$**Activation function**
$$
g(z) = \mathbb{1}_{\{z \ge \theta\}}
$$**Approximation**
$$
\hat{y} = g(f(x)) = \mathbb{1}_{\{ \omega^T x + b \ge \theta\}}
$$
**Perceptron Update Rule**
For each training sample $x_i$:
- $\omega = \omega + \Delta \omega$
- $b = b + \Delta b$
- $\Delta \omega = \alpha \cdot (y_i - \hat{y}) x_i$
- $\Delta b = \alpha \cdot (y_i - \hat{y})$
_explanation:_
$$
\begin{array}{|c|c|c|}
\hline
y & \hat{y} & y - \hat{y} \\
\hline
1 & 1 & 0 \\
\hline
1 & 0 & 1 \\
\hline
0 & 0 & 0 \\
\hline
0 & 1 & -1 \\
\hline
\end{array}
$$
_Weights are pushed towards positive or negative target class in case of missclassification_

### Steps
##### Training (Learn weights):
- Initialize weights
- For each sample:
	- Calculate $\hat{y}$
	- Apply update rule: $\Delta \omega$, $\Delta b$
##### Prediction:
- Calculate $\hat{y} = g(f(x)) = g(\omega^T x + b)$
# SVM

	Use a linear model and try to finad a linear decision boundary (hyperplane) that best separates the data. The best hyperplane is the one that yields the largest separation/margin between both classes. Soo we chose the hyperplane so that the distance from it to the nearest data point on each side is maximized.

**Linear model**
$$
\omega \cdot x_i - b \ge 1 \quad \text{if} \space y_i = 1
$$
$$
\omega \cdot x_i - b \le -1 \quad \text{if} \space y_i = -1
$$
$$
y_i(\omega \cdot x_i - b) \ge 1, \quad y \in \{-1, 1\}
$$
**Loss function: Hinge Loss**
$$
l = 
max(0, 1 - y_i (\omega \cdot x_i - b))
=
\begin{cases} 
0 & \text{if } y \cdot f(x) \geq 1 \\ 
1 - y \cdot f(x) & \text{otherwise} 
\end{cases}
$$**Add Regularization**
$$
J = \lambda ||\omega||^2 +\frac{1}{n} \sum_{i=1}^n max(0, 1 - y_i (\omega \cdot x_i - b)) 
$$
$$
\text{if} \space y_i \cdot f(x_i) \ge 1, \space \text{then} \space J_i = \lambda ||\omega||^2
$$
$$
\text{else:} \space J_i = \lambda ||\omega||^2 + 1 - y_i(\omega \cdot x_i - b)
$$
**Gradients**
- if $y_i \cdot f(x) \ge 1$:
$$
\frac{dJ_i}{d\omega_k} = 2 \lambda \omega_k \quad \quad \frac{dJ_i}{db} = 0
$$
- else:
$$
\frac{dJ_i}{d\omega_k} = 2 \lambda \omega_k - y_i \cdot x_{ik} \quad \quad \frac{dJ_i}{db} = y_i
$$
**Update rule**
- if $y_i \cdot f(x) \ge 1$:
$$
\omega = \omega - \alpha \cdot d\omega = \omega - \alpha \cdot 2 \lambda \omega
$$
$$
b = b - \alpha \cdot db = b
$$
- else:
$$
\omega = \omega - \alpha \cdot d\omega = \omega - \alpha \cdot (2 \lambda \omega - y_i \cdot x_i)
$$
$$
b = b - \alpha \cdot db = b - \alpha \cdot y_i
$$
### Steps
**Training**
- Initialize weights
- Make sure $y \in \{-1, 1\}$
- Apply update rules for n_iters
**Prediction:**
- calculate $y = sign(\omega \cdot x  - b)$
# K-Means

	KMeans is an unsupervised learning method that clusters data set into k different clusters. Each sample is assigned to the cluster with the nearest mean, and then the means (centroids) and clusters are updated during an iterative optimization process.

### Iterative optimization
- Initialize cluster centers (e.g. randomly)
- Repeat until converged:
	- Update cluster labels: Assign points to the nearest cluster center (centroid)
	- Update cluster centers (centroids): Set center to the mean of each cluster
