"""California Housing Price Analysis and Linear Model.

This code loads the California housing dataset and performs three key analyses:
1. Visualizes the relationship between median income and house values
2. Fits a linear model using SVD (Singular Value Decomposition) to predict house values
3. Shows the influence of each feature on house prices through a bar chart visualization

The dataset contains 20,640 observations with 8 features describing California
housing districts including median income, housing age, average rooms, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
housing_data, housing_value = fetch_california_housing(return_X_y=True)
labels = fetch_california_housing().feature_names 

# Plot relationship between median income and house values
plt.scatter(housing_data[:,0], housing_value, s=1)
plt.xlabel('Median Income (x1e4)')
plt.ylabel('Median House Value (x1e5)')
plt.show()

#Normalize the data
housing_data = (housing_data - np.mean(housing_data, axis=0)) / np.std(housing_data, axis=0)

# Prepare data for SVD by adding intercept term
housing_data = np.pad(housing_data, ((0, 0), (0, 1)), mode='constant', constant_values=1)

# Perform SVD and fit linear model
U, S, Vt = np.linalg.svd(housing_data, full_matrices=False)
x = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ housing_value  # Calculate feature coefficients

# Plot actual vs predicted house values
plt.scatter(housing_value, housing_data @ x, s=1)
plt.plot([0, 5], [0, 5], 'k--')  # Diagonal line for reference
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.xlabel('True House Value (x1e5)')
plt.ylabel('Predicted House Value (x1e5)')
plt.show()

# Visualize feature importance
x_tick = range(1, len(x))# Create x-axis positions
plt.bar(x_tick, x[:-1], width=0.5)  # Plot feature influences (excluding intercept)
plt.xlabel('Feature')
plt.ylabel('Influence')
plt.xticks(x_tick, labels, rotation=45)  # Label features on x-axis
plt.show()