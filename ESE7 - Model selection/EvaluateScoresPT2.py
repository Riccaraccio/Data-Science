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

#### WHAT TO DO WITH THE RESULTS? ####
