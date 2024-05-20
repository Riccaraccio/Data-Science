import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets._samples_generator import make_blobs #to generate data clusters

X, Y = make_blobs(n_samples=500, centers=2,
                  random_state=0, cluster_std=0.60)

from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear')
model.fit(X, Y)

# In order to plot the decision boundary, we need to create a grid of points
# Define the range for the grid
x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)

# Create a meshgrid
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Use the model to calculate the decision function over the grid
Z_values = model.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
# ravel() converts a 2D array into a 1D array
# np.c_ concatenates the 1D arrays into a 2D array of shape (n, 2)

# Reshape the result to match the shape of xx and yy
Z_values = Z_values.reshape(xx1.shape)

# Use the model to predict the labels of the data points
Y_pred = model.predict(X)

# Create a scatter plot of the data points
plt.scatter(X[:, 0], X[:, 1], c=Y_pred)

# Add the decision boundary to the plot
plt.contour(xx1, xx2, Z_values, levels=[0], linewidths=2, colors='k')

# Display the plot
plt.show()
