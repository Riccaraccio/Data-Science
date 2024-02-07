import numpy as np
import matplotlib.pyplot as plt

m = -4 #line slope
x = np.linspace(-2, 2, 20) # np.linspace(start, end, n points)
y = m*x +  np.random.randn(x.size) #create random data with random noise

plt.plot(x,m*x, label="True line") # plot true line
plt.plot(x, y, "o", color = "r", label="Noisy data") #plot noisy data

#reshape x to perform SVD, from 1D(20) to 2D(20,1) {-1 refers to w.e. dimension is needed}
x = x.reshape(-1, 1) 
x.reshape(-1,1)

U, S, Vt = np.linalg.svd(x, full_matrices=False) # perfom the SVD
S = np.diag(S) #reconstruction of the S matrix
mtilde  = Vt.T @ np.linalg.inv(S) @ U.T @ y # calculate the approximated shape
plt.plot(x, mtilde*x, "--", label="Regression Line") #plot the regression line

plt.legend() #shows the data label
plt.show()
