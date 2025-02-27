import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.datasets._samples_generator import make_moons #to generate data clusters

X, Y = make_moons(n_samples=500, noise=0.1, random_state=0)

# fit the model
# model = svm.SVC(kernel="linear")
# model = svm.SVC(kernel="poly", degree=2)
model = SVC(kernel="rbf")
model.fit(X, Y)

# In order to plot the decision boundary, we need to create a grid of points
# Define the range for the grid
x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)

# Create a meshgrid
xx1, xx2 = np.meshgrid(x1_range, x2_range)

# Use the model to calculate the decision function over the grid
Z_values = model.decision_function(np.c_[xx1.ravel(), xx2.ravel()])
Z_values = Z_values.reshape(xx1.shape)

plt.contour(xx1, xx2, Z_values, levels=[0], linewidths=2, colors='k')
plt.scatter(X[:, 0], X[:, 1], c=Y) # we use the actual labels to color the points

plt.show()


