import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.datasets._samples_generator import make_moons #to generate data clusters

X, Y = make_moons(n_samples=500, noise=0.1, random_state=0)

# fit the model
# model = svm.SVC(kernel="linear")
# model = svm.SVC(kernel="poly")
model = svm.SVC(kernel="rbf")
model.fit(X, Y)

# Define the range for the grid
x1_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2_range = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)

# Create a meshgrid
xx, yy = np.meshgrid(x1_range, x2_range)

# Use the model to calculate the decision function over the grid
Z_values = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_values = Z_values.reshape(xx.shape)

plt.contour(xx, yy, Z_values, levels=[0], linewidths=2, colors='k')
plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.show()


