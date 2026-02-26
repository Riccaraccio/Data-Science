"""Ridge Regression Regularization Analysis

This script demonstrates the effect of L2 regularization (Ridge regression)
on model coefficients and prediction error using an ill-conditioned matrix.

The code:
1. Creates a Hilbert matrix (known to be numerically unstable)
2. Implements ridge regression with varying regularization strengths
3. Visualizes how coefficients and error change with regularization
4. Shows the bias-variance tradeoff controlled by the lambda parameter

This example illustrates why regularization is important for ill-posed problems
and how it stabilizes the solution by penalizing large coefficient values."
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Create Hilbert matrix as input data (known to be ill-conditioned)
# This matrix has elements X[i,j] = 1/(i+j-1)
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)  # Target vector of all ones

# Create logarithmically spaced regularization parameters (lambda)
n_lambdas = 200
lambdas = np.logspace(-10, -2, n_lambdas)  # From 10^-10 to 10^-2

coefs = []  # Will store coefficient values for each lambda

# Define ridge regression cost function (L2 regularization)
# Minimizes sum of squared errors + alpha * sum of squared weights
def loss(w, X, y, alpha):
    return np.sum((X @ w - y) ** 2) + alpha * np.sum(w ** 2)

mse = []  # Will store mean squared error for each lambda
for l in lambdas:
    # Minimize the ridge regression loss function
    res = minimize(loss, np.zeros(10), args=(X, y, l), tol=1e-7)
    coefs.append(res.x)  # Store the optimal weights
    
    # Compute and store the mean squared error (prediction error)
    mse.append(np.mean((X @ res.x - y) ** 2))

# Alternative implementation using scikit-learn's Ridge (commented out)
# from sklearn.linear_model import Ridge
# for l in lambdas:
#     ridge = Ridge(alpha=l, fit_intercept=False)
#     ridge.fit(X, y)
#     coefs.append(ridge.coef_)

# Create a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# Top subplot: Plot regression coefficients vs regularization strength
ax[0].plot(lambdas, coefs)
ax[0].set_xscale('log')  # Use logarithmic scale for x-axis
ax[0].set_xlabel('lambda')
ax[0].set_ylabel('weights')
ax[0].set_title('Ridge coefficients as a function of the regularization')

# Bottom subplot: Plot mean squared error vs regularization strength
ax[1].plot(lambdas, mse)
ax[1].set_xscale('log')  # Use logarithmic scale for x-axis
ax[1].set_xlabel('lambda')
ax[1].set_ylabel('loss')
ax[1].set_title('Model performance as a function of lambda')

# Improve layout and display the plot
plt.tight_layout()
plt.show()