import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize

# Data
temperature = np.array([11, 10, 10, 9, 9, 9, 
                        10, 12, 13, 15, 17, 19, 
                        22, 24, 25, 24, 23, 21, 
                        20, 17, 15, 14, 13, 12])
time = np.arange(0, 24)

# Function for sinusoidal fit: A*cos(B*t) + C
def sinusoidal_fit(x, time, temperature):
    norm = np.linalg.norm(temperature - (x[0]*np.cos(x[1]*time) + x[2]), ord=2)  
    return norm + 0.1*np.linalg.norm(x, ord=2)

# Create parameter grids
A = np.arange(-10, 10, 1)
B = np.arange(-10, 10, 1)

MSE = np.zeros((len(A), len(B)))

for i in range(len(A)):
    for j in range(len(B)):
        res_sin = scipy.optimize.minimize(sinusoidal_fit, args=(time, temperature), x0=[A[i],B[j],15])    
        x_sin = res_sin.x
        MSE[i,j] = np.linalg.norm(temperature - (x_sin[0]*np.cos(x_sin[1]*time) + x_sin[2]), ord=2)

# Plot contour plot

X, Y = np.meshgrid(A, B)
contour = plt.contourf(X, Y, MSE, cmap='viridis')
plt.colorbar(contour, label='Mean Squared Error')
plt.xlabel('A')
plt.ylabel('B')
plt.title('Normalized MSE of Sinusoidal Fit Across Parameter Space')
plt.show()
 
# Find optimal starting point
# Extract minimum MSE index
min_index = np.unravel_index(np.argmin(MSE, axis=None), MSE.shape)
print(A[min_index[0]],B[min_index[1]])

res_sin = scipy.optimize.minimize(sinusoidal_fit, args=(time, temperature), x0=[A[min_index[0]],B[min_index[1]],15])    
x_sin = res_sin.x
print(x_sin)
# Plot sinusoidal fit
plt.plot(time, x_sin[0]*np.cos(x_sin[1]*time) + x_sin[2], label='Sinusoidal Fit')
plt.scatter(time, temperature)  
plt.show()

# Extract maximum MSE index
max_index = np.unravel_index(np.argmax(MSE, axis=None), MSE.shape)
print(A[max_index[0]],B[max_index[1]])

res_sin = scipy.optimize.minimize(sinusoidal_fit, args=(time, temperature), x0=[A[max_index[0]],B[max_index[1]],15])    
x_sin = res_sin.x
print(x_sin)
# Plot sinusoidal fit
plt.plot(time, x_sin[0]*np.cos(x_sin[1]*time) + x_sin[2], label='Sinusoidal Fit')
plt.scatter(time, temperature)  
plt.show()


