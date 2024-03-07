import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

#load the California housing dataset, returns the data and the target
housing_data, housing_value = fetch_california_housing(return_X_y=True) 

print(housing_data.shape)
print(housing_value.shape)  


plt.scatter(housing_data[:,0],housing_value,s=1)
plt.xlabel('Median Income (x1e4)') #set the x-axis label
plt.ylabel('Median House Value (x1e5)') #set the y-axis label
plt.show()

#pad housing_data with ones for the intercept term
housing_data = np.pad(housing_data, ((0, 0), (0, 1)), mode='constant', constant_values=1)

U, S, Vt = np.linalg.svd(housing_data, full_matrices=False) # perform the SVD
x = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ housing_value # fit parameters influence on the data

plt.plot(housing_value, c="k")
plt.plot(housing_data @ x, c="r")
plt.show()

x_tick = range(len(x)-1)+np.ones(len(x)-1) #create the x-ticks
plt.bar(x_tick, x[:-1], width=0.5) #plot the bar chart
plt.xlabel('Feature') #set the x-axis label
plt.ylabel('Influence') #set the y-axis label
plt.show()

