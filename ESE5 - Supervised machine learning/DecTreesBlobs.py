import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data using make_blobs
X, Y = make_blobs(n_samples=400, centers=3, random_state=0, cluster_std=1.0)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=Y)

# Split the data into a training set and a test set
X_train = X[:300,:]
Y_train = Y[:300]

X_test = X[300:,:]
Y_test = Y[300:]

from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Train the decision tree classifier on the training set
model.fit(X_train, Y_train)

# Define the range for the grid
x1_range = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
x2_range = np.linspace(np.min(X_train[:, 1]), np.max(X_train[:, 1]), 100)

# Create a meshgrid
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Use the trained model to calculate the decision function over the grid
Z_values = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z_values = Z_values.reshape(xx1.shape)

# Plot the decision boundaries
plt.contourf(xx1, xx2, Z_values, alpha=0.3)
plt.show()

# Use the trained model to predict the labels of the test set
Y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate the confusion matrix
mat = confusion_matrix(Y_test, Y_pred)

# Plot the confusion matrix
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False) #annot=True to annotate cells, fmt='d' to disable scientific notation
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
