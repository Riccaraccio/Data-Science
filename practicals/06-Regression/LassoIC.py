"""Lasso Regression with Information Criteria for Feature Selection.

This code performs Lasso regression on the diabetes dataset with:
1. Loading the diabetes dataset and adding random features
2. Standardizing the features for optimal performance
3. Training Lasso models with varying regularization strengths
4. Evaluating models using AIC and BIC information criteria
5. Visualizing the impact of alpha parameter on model selection

The program creates a visualization showing how different alpha values affect
the information criteria scores, which helps identify the optimal level of 
regularization. Lasso regression performs both regularization and feature selection
by shrinking less important feature coefficients to zero.

Data Structure
-------------
- X: Standardized feature matrix (including random features)
- y: Target diabetes progression values
- alphas: Range of regularization parameters to evaluate
- scores_aic: AIC scores for each alpha value
- scores_bic: BIC scores for each alpha value
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)

# Add random features to make feature selection more relevant
rng = np.random.RandomState(42)
n_random_features = 14
X_random = rng.randn(X.shape[0], n_random_features)
X = np.c_[X, X_random]

# Standardize features for optimal performance
X_avg = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_avg) / X_std

# Define information criteria functions
def AIC(y_true, y_pred, k):
    """Calculate Akaike Information Criterion.
    
    Parameters:
    y_true: Actual target values
    y_pred: Predicted target values
    k: Number of parameters in the model
    
    Returns:
    AIC score (lower is better)
    """
    n = len(y_true)
    residual_sum_of_squares = ((y_true - y_pred) ** 2).sum()
    aic = n * np.log(residual_sum_of_squares / n) + 2 * k
    return aic

def BIC(y_true, y_pred, k):
    """Calculate Bayesian Information Criterion.
    
    Parameters:
    y_true: Actual target values
    y_pred: Predicted target values
    k: Number of parameters in the model
    
    Returns:
    BIC score (lower is better)
    """
    n = len(y_true)
    residual_sum_of_squares = ((y_true - y_pred) ** 2).sum()
    bic = n * np.log(residual_sum_of_squares / n) + k * np.log(n)
    return bic

# Prepare to collect scores
scores_aic = []
scores_bic = []

# Create range of alpha values to test
alphas = np.logspace(-2, 2, 100)

# Train models and calculate information criteria
for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X, y)
    y_pred = lasso.predict(X)
    
    # Count non-zero coefficients (effective parameters)
    k = np.sum(lasso.coef_ != 0) + 1  # Number of features + intercept
    
    # Calculate and store information criteria
    scores_aic.append(AIC(y, y_pred, k))
    scores_bic.append(BIC(y, y_pred, k))

# Create visualization of scores
plt.figure(figsize=(12, 6))
plt.plot(alphas, scores_aic, label='AIC', color='blue')
plt.plot(alphas, scores_bic, label='BIC', color='red')
plt.plot(alphas, np.sqrt([scores_aic[i] * scores_bic[i] for i in range(len(alphas))]), 
         label='Geometric Mean (AIC*BIC)^0.5', color='green')
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Information Criteria Score')
plt.title('AIC and BIC Scores for Lasso Regression')
plt.legend()
plt.grid()
plt.show()