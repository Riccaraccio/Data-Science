import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the Rössler system
def rossler(X, t, a, b, c):
    x, y, z = X
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# Parameters
a, b, c = 0.2, 0.2, 5.7
X0 = [0.0, 1.0, 0.0]  # Initial condition
t = np.linspace(0, 100, 10000)

# Integrate the Rössler equations
X = odeint(rossler, X0, t, args=(a, b, c))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:,0], X[:,1], X[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Save the data as npy array

np.save('RosslerAttractor.npy', X)