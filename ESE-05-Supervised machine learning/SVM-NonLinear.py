"""Non-Linear Classification of Moon-Shaped Data using SVM.

This code performs binary classification on non-linearly separable data by:
1. Generating synthetic moon-shaped data clusters with noise
2. Training a Support Vector Classifier with an RBF kernel
3. Creating a decision boundary visualization
4. Displaying the classification results with original labels

The program visualizes how a radial basis function (RBF) kernel enables
the SVM to find a non-linear decision boundary that separates the
two interleaved moon-shaped clusters.

Data Structure
-------------
- X: 2D coordinates of data points
- Y: Binary labels for each data point
- Z_values: Decision function values over a grid for boundary plotting
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets._samples_generator import make_moons  # to generate data clusters

# Generate synthetic moon-shaped data with noise
X, Y = make_moons(n_samples=500, noise=0.1, random_state=0)

# Train the model with a radial basis function kernel
# Other commented options include linear and polynomial kernels
# model = svm.SVC(kernel="linear")
# model = svm.SVC(kernel="poly", degree=2)
model = SVC(kernel="rbf")
model.fit(X, Y)  # Train the SVM model on the generated data

# In order to plot the decision boundary, we need to create a grid of points
# Define the range for the grid
x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)

# Create a meshgrid for evaluation
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Use the model to calculate the decision function over the grid
Z_values = model.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
Z_values = Z_values.reshape(xx1.shape)

# Plot the decision boundary as a contour line
plt.contour(xx1, xx2, Z_values, levels=[0], linewidths=2, colors='k')

# Create a scatter plot of the data points colored by their original labels
plt.scatter(X[:, 0], X[:, 1], c=Y)  # we use the actual labels to color the points

# Display the plot
plt.show()