import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = np.load('RosslerAttractor.npy')
print("Data shape:", X.shape)

# # add noise to the data
# noise_level = 0.05
# X = X + noise_level * np.random.randn(*X.shape)

# Plot the 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

"""# Animate the 3D trajectory
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(0, 20)

def animate(i):
    line.set_data(X[:i, 0], X[:i, 1])
    line.set_3d_properties(X[:i, 2])
    return line,
# Increase speed by reducing interval and increasing step
interval = 0.01  # Reduce interval to make it faster
frame_step = 5   # Increase frame step to skip frames
ani = FuncAnimation(fig, animate, frames=range(0, len(X), frame_step), interval=interval, blit=True)

plt.show()"""

# Compute the time derivatives
dt = 0.01
dX = (X[1:] - X[:-1]) / dt

# Construct the library
# We will use a library of monomials up to degree 2

# Extract individual components
x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

# Compute the additional monomials
ones = np.ones_like(x)
xy = x * y
xz = x * z
yz = y * z
x2 = x ** 2
y2 = y ** 2
z2 = z ** 2

# Stack all the components together
theta = np.column_stack((ones, x, y, z, xy, xz, yz, x2, y2, z2))

theta = theta[:-1] # Remove the last row to match the size of dX

def SINDy(theta, dXdt, lambd, n):
    #Inital guess   
    Xi = np.linalg.lstsq(theta, dXdt)[0] # lstsq returns a tuple, we only need the first element 
    
    for k in range(10): # iteratively sparsify the solution
        smallinds = np.abs(Xi) < lambd # find the index of coefficients smaller than the threshold, return a boolean matrix
        Xi[smallinds] = 0 # set the small coefficients to zero
        
        for var in range(n): # n is the number of state variables
            biginds = smallinds[:,var] == 0 # find the index of coefficients larger than the threshold
            Xi[biginds,var] = np.linalg.lstsq(theta[:,biginds], dXdt[:,var])[0] # regress the dynamics of the state variables on the big coefficients
    
    return Xi

# Set the threshold
lambd = 0.1

# Compute the coefficients
Xi = SINDy(theta, dX, lambd, n=3)

# Print the coefficients
print(Xi)

