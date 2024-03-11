import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import scipy.optimize

def count_non_zero(array):
    count = 0
    for num in array:
        if num != 0:
            count += 1
    return count

def calculate_aic(beta, sigma_sq, y_true, x):
    n = len(y_true)
    residuals = y_true - np.dot(x, beta)
    n_params = count_non_zero(beta)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals ** 2) / sigma_sq

    aic =  2*n_params - 2*log_likelihood
    return aic

def calculate_bic(beta, sigma_sq, y_true, x):
    n = len(y_true)
    residuals = y_true - np.dot(x, beta)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_sq) - 0.5 * np.sum(residuals ** 2) / sigma_sq
    n_params = count_non_zero(beta)
    bic =  np.log(n)*n_params - 2*log_likelihood
    return bic

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Pad X with ones for the intercept term
X = np.pad(X, ((0, 0), (0, 1)), mode='constant', constant_values=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def model_construction(X, y) -> np.ndarray:
    
    # YOUR CODE HERE ########################################
    
    
    return np.zeros(X.shape[1])
    
    #########################################################

# Train the model
beta = model_construction(X_train, y_train)

if beta.shape != (X_train.shape[1],):
    raise ValueError("The shape of x_parameters is incorrect.")

print("Your model parameters are: ",beta)

print("Number of non-zero parameter:", count_non_zero(beta))

sigma_sq = mean_squared_error(y_train, X_train @ beta)

model_aic= calculate_aic(beta, sigma_sq, y_test, X_test)
model_bic = calculate_bic(beta, sigma_sq, y_test, X_test)

print("The AIC of your model is: {:.4f}".format(model_aic))
print("The BIC of your model is: {:.4f}".format(model_bic))

model_overall_score = model_aic + model_bic
print("The overall score of your model is: {:.6f}. Do you think you can do better?".format(model_overall_score))

# plt.plot(range(X_test.shape[0]), y_test, label="True values")
# plt.plot(range(X_test.shape[0]), X_test@ beta, label="True values")

plt.bar(range(1,beta.shape[0]+1), beta, label="Coefficients")
# plt.scatter(X_test @ beta, y_test)
# plt.plot([0, 350], [0, 350], 'r')
# plt.xlabel("Predicted values")
# plt.ylabel("True values")
# plt.axis('square')
# plt.xlim(0, 350)
# plt.ylim(0, 350)
plt.show()  
