import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California housing dataset
housing_data, housing_value = fetch_california_housing(return_X_y=True)

housing_data = housing_data[:150]
housing_value = housing_value[:150]
# fig, ax = plt.subplots(1,3)
# for i in range(3):
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(housing_data, housing_value, test_size=0.3, random_state=i)

#     # perform pinv regression
#     beta_pinv = np.linalg.pinv(X_train) @ y_train

#     ax[i].bar(range(len(beta_pinv)), beta_pinv)
# plt.show()

# perform k fold cross validation
values = (1, 10, 100)
fig, ax = plt.subplots(2,3)

for i in range(len(values)):
    beta_pinv = np.zeros((housing_data.shape[1],))
    beta_lasso = np.zeros((housing_data.shape[1],))
    
    for j in range(values[i]):
        X_train, X_test, y_train, y_test = train_test_split(housing_data, housing_value, test_size=0.3, random_state=j)
        
        # PINV regression
        beta_pinv += np.linalg.pinv(X_train) @ y_train
        
        # Lasso regression
        lasso = linear_model.Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        beta_lasso += lasso.coef_
        
    beta_pinv /= values[i]
    beta_lasso /= values[i]
    
    ax[0,i].bar(range(len(beta_pinv)), beta_pinv)
    ax[0,i].set_title('PINV {}'.format(values[i]))
    ax[1,i].bar(range(len(beta_lasso)), beta_lasso)
    ax[1,i].set_title('Lasso {}'.format(values[i]))
    
plt.show()