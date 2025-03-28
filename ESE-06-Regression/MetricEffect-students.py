"""Ridge Regression Regularization Analysis

This script demonstrates the effect of L2 regularization (Ridge regression)
on model coefficients and prediction error using an ill-conditioned matrix.

The code:
1. Creates a Hilbert matrix (known to be numerically unstable)
2. Implements ridge regression with varying regularization strengths
3. Visualizes how coefficients and error change with regularization
4. Shows the bias-variance tradeoff controlled by the lambda parameter
"""

# Add the necessary imports
# import ...

# Create Hilbert matrix as input data (known to be ill-conditioned)
# This matrix has elements X[i,j] = 1/(i+j+1)
# X = ...

# Target vector of all ones
# y = ...  

# Create logarithmically spaced regularization parameters (lambda)
n_lambdas = 200
# lambdas = ...  # From 10^-10 to 10^-2

coefs = []  # Will store coefficient values for each lambda

# Define ridge regression cost function (L2 regularization)
# Minimizes sum of squared errors + alpha * sum of squared weights
# def loss(w, X, y, alpha):
    #return ...

mse = []  # Will store mean squared error for each lambda
#for l in lambdas:
    # Minimize the ridge regression loss function
    # Use scipy's minimize function to find the optimal weights, set tolerance to 1e-7: 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    # res = ...
    
    # Store the optimal weights
    
    # Compute and store the mean squared error (prediction error) and append to mse list


# Create a figure with two subplots
# fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# Top subplot: Plot regression coefficients vs regularization strength
# ax[0].plot(lambdas, coefs)
# ax[0].set_xscale('log')  # Use logarithmic scale for x-axis
# ax[0].set_xlabel('lambda')
# ax[0].set_ylabel('weights')
# ax[0].set_title('Ridge coefficients as a function of the regularization')

# Bottom subplot: Plot mean squared error vs regularization strength
# ax[1].plot(lambdas, mse)
# ax[1].set_xscale('log')  # Use logarithmic scale for x-axis
# ax[1].set_xlabel('lambda')
# ax[1].set_ylabel('loss')
# ax[1].set_title('Model performance as a function of lambda')

# Improve layout and display the plot
# plt.tight_layout()
# plt.show()