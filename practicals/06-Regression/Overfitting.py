"""Polynomial Regression with Regularization for Model Complexity Analysis.

This code demonstrates polynomial regression with varying degrees of complexity:
1. Generating synthetic quadratic data with controlled noise
2. Splitting data into training and test sets
3. Fitting polynomial models of increasing complexity (degrees 1-14)
4. Comparing standard Linear Regression vs Lasso regularization
5. Analyzing overfitting through training and testing error visualization

The program creates visualizations showing how polynomial degree affects model fit
and error rates. Lasso regression is used to demonstrate how regularization can
help control model complexity and reduce overfitting in high-degree polynomials.

Data Structure
-------------
- x: Independent variable values (synthetic data)
- y_true: Actual underlying quadratic relationship (2 + 3x - 0.5x²)
- y_noisy: Target values with added Gaussian noise
- X_train, X_test: Training and test feature sets
- y_train, y_test: Training and test target values
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data points along x-axis
x = np.linspace(0, 10, 40)

# Create true underlying quadratic relationship: y = 2 + 3x - 0.5x²
y_true = 2 + 3*x - 0.5*x**2

# Add Gaussian noise to create realistic data
noise_level = 5
y_noisy = y_true + np.random.normal(0, noise_level, size=len(x))

# Split data into training and testing sets (70% train, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y_noisy, test_size=0.3, random_state=42)

# Visualize the original data with training/testing split
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, label='Testing Data', color='orange')
plt.plot(x, y_true, label='True Relationship', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Visualization')
plt.legend()
plt.show()

# Define polynomial degrees to evaluate, from linear to degree 14
degrees = range(1, 15)  # From linear (degree 1) to degree 14

# Initialize lists to store error metrics
linear_train_errors = []
linear_test_errors = []
lasso_train_errors = []
lasso_test_errors = []

# Import regression models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# Evaluate models of increasing polynomial complexity
for degree in degrees:
    # Create polynomial features using Vandermonde matrix
    # Vandermonde matrix creates columns for [1, x, x², x³, ...] up to specified degree
    X_train_poly = np.vander(X_train, degree + 1, increasing=True)
    X_test_poly = np.vander(X_test, degree + 1, increasing=True)

    # Fit standard polynomial regression model (no regularization)
    model = LinearRegression()
    
    # Fit Lasso model with L1 regularization (promotes sparsity in coefficients)
    # alpha=1 controls regularization strength, max_iter ensures convergence
    model_lasso = Lasso(alpha=1, max_iter=10000)

    # Train both models
    model_lasso.fit(X_train_poly, y_train)
    model.fit(X_train_poly, y_train)

    # Generate predictions for standard linear model
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculate and store mean squared errors for linear model
    linear_train_errors.append(np.mean((y_train - y_train_pred) ** 2))
    linear_test_errors.append(np.mean((y_test - y_test_pred) ** 2))

    # Visualize curve fitting for selected polynomial degrees
    if degree in [2, 4, 8, 14]:
        plot_y = model.predict(np.vander(x, degree + 1, increasing=True))
        plt.plot(x, plot_y, label=f'Degree {degree}')

    # Generate predictions for Lasso model
    y_train_pred = model_lasso.predict(X_train_poly)
    y_test_pred = model_lasso.predict(X_test_poly)

    # Calculate and store mean squared errors for Lasso model
    lasso_train_errors.append(np.mean((y_train - y_train_pred) ** 2))
    lasso_test_errors.append(np.mean((y_test - y_test_pred) ** 2))

# Set plot boundaries for better visualization
plt.xlim(0, 10)
plt.ylim(-25, 15)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Models')
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, label='Testing Data', color='orange')
plt.plot(x, y_true, label='True Relationship', color='red')
plt.legend()        
plt.show()

# Plot training and testing errors to visualize overfitting
plt.figure(figsize=(10, 6))
plt.plot(degrees, linear_train_errors, label='Training Error', marker='o')
plt.plot(degrees, lasso_train_errors, label='Lasso Training Error', marker='o')
plt.plot(degrees, linear_test_errors, label='Testing Error', marker='o')
plt.plot(degrees, lasso_test_errors, label='Lasso Testing Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Testing Error')
plt.legend()
plt.yscale('log')  # Log scale helps visualize error differences
plt.grid(True)
plt.show()