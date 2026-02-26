"""Comparison of L2 (Least Squares) and Lasso Regression.

This code demonstrates the difference between L2 and Lasso regression
in recovering sparse signals from noisy measurements. It shows how
Lasso regression can better recover sparse solutions compared to
standard least squares regression.

The setup involves:
1. Creating a sparse signal (x) with only two non-zero components
2. Generating noisy measurements (b) using a random measurement matrix (A)
3. Attempting to recover x using both L2 and Lasso regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Generate synthetic data
np.random.seed(0)  # For reproducibility
A = np.random.randn(100,10)  # Random measurement matrix
x = np.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 0])  # True sparse signal
b = A @ x + np.random.randn(100)  # Noisy measurements

# Perform L2 regression (least squares)
xL2 = np.linalg.pinv(A) @ b  # Using pseudoinverse
print(xL2)  # Show L2 solution

# Perform Lasso regression
reg = linear_model.Lasso(alpha=0.2).fit(A, b)
xLasso = reg.coef_  # Get Lasso coefficients
print(xLasso)

# Visualize results with bar plot
width = 0.2
positions = np.arange(len(x))

# Create comparative bar plot in one go with fixed offsets
plt.bar(positions - width, x, width=width, label='True x')
plt.bar(positions, xL2, width=width, label='X regressed with pinv')
plt.bar(positions + width, xLasso, width=width, label='X regressed with Lasso')

plt.set_cmap("jet")
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot of Vectors')
plt.legend()
plt.show()