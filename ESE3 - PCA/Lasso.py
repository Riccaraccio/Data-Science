import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection

np.random.seed(0) #set the seed for the random number generator
A = np.random.randn(100,10) # Matrix of possible predictors
x = np.array([0, 0, 1, 0, 0, 0, -1, 0, 0, 0]) #Two nonzero predictors out of 10

b = A @ x + 2*np.random.randn(100) #A*x + random noise(dimension = 100)

xL2 = np.linalg.pinv(A) @ b #least square regression

print(xL2) # should return x but doesnt 

# Lasso regression
reg = linear_model.Lasso(alpha=0.2).fit(A, b)

xLasso = reg.coef_ #get the lasso coefficent x
print(xLasso)

# Set the width of the bars
bar_width = 0.2

# Create positions for the bars
bar_positions_x = np.arange(len(x))
bar_positions_xL2 = bar_positions_x + bar_width
bar_positions_xLasso = bar_positions_xL2 + bar_width

# Create bar plots for each vector
plt.bar(bar_positions_x, x, width=bar_width, label='True x')
plt.bar(bar_positions_xL2, xL2, width=bar_width, label='X regressed with pinv')
plt.bar(bar_positions_xLasso, xLasso, width=bar_width, label='X regressed with Lasso')

# Add labels and title
plt.set_cmap("jet")
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot of Vectors')

plt.legend()

# Show the plot
plt.show()
