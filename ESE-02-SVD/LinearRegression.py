"""Linear Regression with SVD and Outlier Effect Demonstration.

This code demonstrates linear regression using SVD (Singular Value Decomposition)
on noisy data. The true relationship between x and y is y = mx, where m = -4.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate data with known slope and random noise
m = -4  # true line slope
x = np.linspace(-2, 2, 20)  # create evenly spaced x values
y = m*x + np.random.randn(x.size)  # add random Gaussian noise to true line

# Plot initial data and true line
plt.plot(x, m*x, color="black", label="True line")
plt.plot(x, y, "o", color="red", label="Noisy data")

# Prepare data for SVD
x = x.reshape(-1, 1)  # reshape x to 2D array for SVD

# Perform SVD and calculate regression line
U, S, Vt = np.linalg.svd(x, full_matrices=False)
S = np.diag(S)  # convert singular values to diagonal matrix
mtilde = Vt.T @ np.linalg.inv(S) @ U.T @ y  # compute slope using SVD
plt.plot(x, mtilde*x, "--", label="Regression Line")

plt.legend()
plt.show()