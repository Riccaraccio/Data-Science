import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd

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

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.3, random_state=0)

df = pd.read_excel('DATA SCIENCE _linear regression cup.xlsx')   
df = df.rename(columns={"Beta parameters": 'beta', 'Declare your overall score': 'score'})

for index, student in df.iterrows():
    
    # Convert the elements to float and create a NumPy array
    student["beta"] = student["beta"] .replace('[', '').replace(']', '')
    beta_elements = student["beta"].split()
    beta = np.array([float(element) for element in beta_elements])
    student["beta"] = beta
    # Compute the number of non-zero parameters
    student["n_params"] = count_non_zero(student["beta"])
    
    # predict y
    y_pred = np.dot(X_test, student["beta"])
    
    # Compute the mean squared error
    student["mse"] = mean_squared_error(y_test, y_pred)
    
    # Compute the AIC and BIC
    student["aic"] = calculate_aic(student["beta"], student["mse"], y_test, X_test)
    student["bic"] = calculate_bic(student["beta"], student["mse"], y_test, X_test)
    student["real score"] = student["aic"] + student["bic"] 

    # Update the DataFrame
    df.loc[index, "real score"] = student["real score"]

# Sort the DataFrame by the real score
df = df.sort_values(by="real score")

# Select the top n students
n = 2  
top_students = df.head(n)

# Create a new DataFrame with selected columns
results_df = top_students[["Nome", "real score", "beta"]]

# plot the top n students scores
plt.bar(range(1, n+1), results_df["real score"])
plt.title("Score of the top {} students".format(n))
plt.ylabel("Score")
plt.xticks(range(1, n+1))

for i in range(1, n+1):
    plt.text(i, results_df.loc[i-1,"real score"] + 0.5, results_df.loc[i-1,"Nome"], ha='center')
plt.ylim(min(results_df["real score"]) - 1, max(results_df["real score"]) + 1)
plt.show()


# COMPARE BETA PARAMETERS
fig, ax = plt.subplots(1, n)
# Initialize variables to store min and max values of beta across all students
min_beta = np.inf
max_beta = -np.inf

for index, student in results_df.iterrows():
    # Convert the elements to float and create a NumPy array
    student["beta"] = student["beta"].replace('[', '').replace(']', '')
    beta_elements = student["beta"].split()
    beta = np.array([float(element) for element in beta_elements])
    student["beta"] = beta
    
    # Update min and max beta values
    min_beta = min(min_beta, np.min(beta))
    max_beta = max(max_beta, np.max(beta))
    
    ax[index].bar(range(len(student["beta"])), student["beta"])
    ax[index].set_title(student["Nome"])
    
for ax in ax:
    ax.set_ylim(min_beta, max_beta)
    
plt.show()


# COMPARE MODEL PERFORMANCE
fig, ax = plt.subplots(1, n)

for index, student in results_df.iterrows():
    # Convert the elements to float and create a NumPy array
    student["beta"] = student["beta"].replace('[', '').replace(']', '')
    beta_elements = student["beta"].split()
    beta = np.array([float(element) for element in beta_elements])
    student["beta"] = beta
    
    # predict y
    y_pred = np.dot(X_test, student["beta"])
    
    ax[index].scatter(y_pred, y_test)
    ax[index].plot([0, 350], [0, 350], 'r')
    ax[index].set_title(student["Nome"])
    ax[index].set_xlabel("Predicted values")
    ax[index].set_ylabel("True values")
    ax[index].axis('equal')
    
plt.show()