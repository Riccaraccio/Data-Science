import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

#load the California housing dataset, returns the data and the target
housing_data, housing_value = fetch_california_housing(return_X_y=True) 

print(housing_data.shape)
print(housing_value.shape)  


plt.scatter(housing_data[:,3],housing_value,s=1)
plt.xlabel('Median Income (x10000)') #set the x-axis label
plt.ylabel('Median House Value (x100000)') #set the y-axis label
plt.show()

# housing_data = housing_data - np.mean(housing_data, axis=0) #center the data

U, S, Vt = np.linalg.svd(housing_data, full_matrices=False) # perform the SVD
x = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ housing_value # calculate the approximated shape

x_tick = range(len(x)-1)+np.ones(len(x)-1) #create the x-ticks
plt.bar(x_tick, x[:-1], width=0.5) #plot the bar chart
plt.show()

