import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns

# Generate synthetic data using make_blobs
X, Y = make_blobs(n_samples=400, centers=4, random_state=0, cluster_std=1.0)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=Y)

# Split the data into a training set and a test set
X_fit = X[:300, :]
Y_fit = Y[:300]

X_cv = X[300:, :]
Y_cv = Y[300:]

from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
tree = DecisionTreeClassifier()

# Train the decision tree classifier on the training set
tree.fit(X_fit, Y_fit)

# Define the range for the grid
x1_range = np.linspace(np.min(X_fit[:, 0]), np.max(X_fit[:, 0]), 100)
x2_range = np.linspace(np.min(X_fit[:, 1]), np.max(X_fit[:, 1]), 100)

# Create a meshgrid
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Use the trained model to calculate the decision function over the grid
Z_values = tree.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z_values = Z_values.reshape(xx1.shape)

# Plot the decision boundaries
plt.contourf(xx1, xx2, Z_values, alpha=0.3)
plt.show()

# Use the trained model to predict the labels of the test set
Y_cv_pred = tree.predict(X_cv)

from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
mat = confusion_matrix(Y_cv, Y_cv_pred)

# Plot the confusion matrix
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False) #annot=True to annotate cells, fmt='d' to disable scientific notation
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
