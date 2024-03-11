import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge

housing_data, housing_values = fetch_california_housing(return_X_y=True)

# PINV regression
beta_pinv = np.linalg.pinv(housing_data) @ housing_values
# Description: PINV regression uses the Moore-Penrose pseudo-inverse to calculate the coefficients of the linear regression model.

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(housing_data, housing_values)
beta_lasso = lasso.coef_
# Description: Lasso regression is a linear regression method that performs both feature selection and regularization by adding a penalty term to the loss function.

# Ridge regression
ridge = Ridge(alpha=0.1)
ridge.fit(housing_data, housing_values)
beta_ridge = ridge.coef_
# Description: Ridge regression is a linear regression method that adds a penalty term to the loss function to prevent overfitting and reduce the impact of multicollinearity.

# Roubust fit regression
ransac = RANSACRegressor()
ransac.fit(housing_data, housing_values)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
beta_ransac = ransac.estimator_.coef_
# Description: RANSAC (RANdom SAmple Consensus) regression is a robust regression method that iteratively fits the model to a subset of inlier data points, ignoring outliers.

# elastic net regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(housing_data, housing_values)
beta_elastic = elastic.coef_
# Description: Elastic Net regression is a linear regression method that combines the L1 and L2 regularization penalties of Lasso and Ridge regression, respectively.

# Bayesian regression
bayesian = BayesianRidge()
bayesian.fit(housing_data, housing_values)
beta_bayesian = bayesian.coef_
# Description: Bayesian regression is a linear regression method that incorporates prior knowledge about the coefficients into the model estimation, resulting in more robust and interpretable results.


# Plot the coefficients 
fig, ax = plt.subplots(2,3, figsize=(10,10))
ax[0,0].bar(range(len(beta_pinv)), beta_pinv)
ax[0,0].set_title('PINV')
ax[0,1].bar(range(len(beta_lasso)), beta_lasso)
ax[0,1].set_title('Lasso')
ax[1,0].bar(range(len(beta_ridge)), beta_ridge)
ax[1,0].set_title('Ridge')
ax[1,1].bar(range(len(beta_ransac)), beta_ransac)
ax[1,1].set_title('RANSAC')
ax[0,2].bar(range(len(beta_elastic)), beta_elastic)
ax[0,2].set_title('Elastic Net')
ax[1,2].bar(range(len(beta_bayesian)), beta_bayesian)
ax[1,2].set_title('Bayesian')

plt.show()
