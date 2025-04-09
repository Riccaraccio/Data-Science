# Add the necessary imports
import numpy as np
# import ...

# Set random seed for reproducibility
np.random.seed(42)

# Create x vector from 0 to 10
# x = ...

# Create y_true vector using a quadratic function
# y = 2 + 3x - 1/2 x^2
# y_true = ...

noise_level = 5

# Add noise to the y_true vector using np.random.normal
# y_noisy = y_true + ...

# Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(..., test_size=0.3, random_state=42)

# Visualize the data
# plt.scatter(X_train, y_train, label='Training Data')
# plt.scatter(X_test, y_test, label='Testing Data', color='orange')
# plt.plot(x, y_true, label='True Relationship', color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data Visualization')
# plt.legend()
# plt.show()

degrees = range(1, 15)  # From linear (degree 1) to degree 14

# vectors to store training and testing errors
linear_train_errors = []
linear_test_errors = []

#iterate over polynomial degrees
# for degree in ... :
    # create polynomial features
    # HINT: check https://numpy.org/doc/stable/reference/generated/numpy.vander.html
    # set increasing=True to get the polynomial in increasing order
    # X_train_poly = ...
    # X_test_poly = ...

    # Fit polynomial regression model
    # HINT: check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # model = ...

    # Fit the data

    # Predict train and test label 
    # y_train_pred = ...
    # y_test_pred = ...

    # Calculate training and testing errors and append to the lists


# Plot training and testing errors
# Plot degrees vs training and test errors
# use logarithmic scale on y-axis
