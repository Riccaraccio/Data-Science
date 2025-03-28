import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

x = np.linspace(0, 10, 40)
y_true = 2 + 3*x - 0.5*x**2

noise_level = 5
y_noisy = y_true + np.random.normal(0, noise_level, size=len(x))

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y_noisy, test_size=0.3, random_state=42)

# Visualize the data
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, label='Testing Data', color='orange')
plt.plot(x, y_true, label='True Relationship', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Visualization')
plt.legend()
plt.show()

degrees = range(1, 15)  # From linear (degree 1) to degree 14

linear_train_errors = []
linear_test_errors = []
lasso_train_errors = []
lasso_test_errors = []

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
for degree in degrees:
    # create polynomial features
    X_train_poly = np.vander(X_train, degree + 1, increasing=True)
    X_test_poly = np.vander(X_test, degree + 1, increasing=True)

    # Fit polynomial regression model
    model = LinearRegression()
    # Fit lasso model
    model_lasso = Lasso(alpha=1, max_iter=10000)

    model_lasso.fit(X_train_poly, y_train)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    linear_train_errors.append(np.mean((y_train - y_train_pred) ** 2))
    linear_test_errors.append(np.mean((y_test - y_test_pred) ** 2))

    if degree in [2, 4, 8, 14]:
        plot_y = model.predict(np.vander(x, degree + 1, increasing=True))
        plt.plot(x, plot_y, label=f'Degree {degree}')

    y_train_pred = model_lasso.predict(X_train_poly)
    y_test_pred = model_lasso.predict(X_test_poly)

    lasso_train_errors.append(np.mean((y_train - y_train_pred) ** 2))
    lasso_test_errors.append(np.mean((y_test - y_test_pred) ** 2))

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

# Plot training and testing errors
plt.figure(figsize=(10, 6))
plt.plot(degrees, linear_train_errors, label='Training Error', marker='o')
plt.plot(degrees, lasso_train_errors, label='Lasso Training Error', marker='o')
plt.plot(degrees, linear_test_errors, label='Testing Error', marker='o')
plt.plot(degrees, lasso_test_errors, label='Lasso Testing Error', marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Training vs Testing Error')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

