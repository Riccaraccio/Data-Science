"""Rossler Attractor Analysis and SINDy for System Identification

This code performs analysis and system identification on a Rossler Attractor dataset by:

Loading and visualizing the attractor trajectory in 3D
Creating visualizations of the system dynamics at different time points
Computing time derivatives of the state variables
Constructing a library of polynomial functions for SINDy modeling
Implementing the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm
Identifying and printing the governing equations of the system

The program displays visualizations of the 3D attractor trajectory, snapshots showing
the evolution of the system over time, and the identified sparse coefficients that
represent the underlying dynamics. SINDy is used to discover the governing equations
directly from data, enabling model discovery with interpretable mathematical expressions.

Data Structure:
X: Original 3D trajectory data of the Rossler system
dX: Time derivatives of the state variables
theta: Library of candidate functions (polynomial terms)
Xi: Sparse coefficient matrix representing the discovered model
"""

# Import required libraries for numerical computation, visualization, and file operations
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("ESE-11-Physics informed models")  # Change to the directory containing the dataset

# Load the Rossler attractor trajectory data from a NumPy file
X = np.load('RosslerAttractor.npy')
print("Data shape:", X.shape)  # Print the dimensions of the loaded data

# Code for adding artificial noise to test algorithm robustness (currently disabled)
# add noise to the data
noise_level = 0.05  # Define the relative amplitude of noise to add
X = X + noise_level * np.random.randn(*X.shape)  # Add Gaussian noise to all data points

# Create a 3D visualization of the complete Rossler attractor trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
ax.plot(X[:, 0], X[:, 1], X[:, 2])  # Plot the 3D trajectory using x, y, z coordinates
ax.set_xlabel('X')  # Label the x-axis
ax.set_ylabel('Y')  # Label the y-axis
ax.set_zlabel('Z')  # Label the z-axis
plt.show()  # Display the static 3D plot

# Create an animation to visualize the temporal evolution of the system
from matplotlib.animation import FuncAnimation  # Import animation capabilities

# Set up the animation figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [])  # Initialize an empty line object for animation
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-10, 10)  # Set x-axis limits for better visualization
ax.set_ylim(-10, 10)  # Set y-axis limits for better visualization
ax.set_zlim(0, 20)    # Set z-axis limits for better visualization

def animate(i):
    """Update function for animation - draws the trajectory up to frame i"""
    line.set_data(X[:i, 0], X[:i, 1])  # Update x and y coordinates
    line.set_3d_properties(X[:i, 2])   # Update z coordinates
    return line,

# Animation performance optimization parameters
interval = 0.01    # Time interval between frames (in milliseconds) - smaller means faster
frame_step = 5     # Number of data points to skip between frames - larger means faster
ani = FuncAnimation(fig, animate, frames=range(0, len(X), frame_step), 
                   interval=interval, blit=True)  # Create the animation

plt.show()  # Display the animation

# Compute the time derivatives of state variables using finite differences
dt = 0.01  # Time step between consecutive data points
dX = (X[1:] - X[:-1]) / dt  # Forward difference approximation of derivatives

# Construct the library of candidate functions for SINDy
# Extract individual state variables from the trajectory data
x = X[:, 0]  # x-coordinate time series
y = X[:, 1]  # y-coordinate time series
z = X[:, 2]  # z-coordinate time series

# Compute polynomial terms up to degree 2 for the function library
ones = np.ones_like(x)  # Constant term (1)
xy = x * y    # Cross term (xy)
xz = x * z    # Cross term (xz)
yz = y * z    # Cross term (yz)
x2 = x ** 2   # Quadratic term (x^2)
y2 = y ** 2   # Quadratic term (y^2)
z2 = z ** 2   # Quadratic term (z^2)

# Combine all candidate functions into a single library matrix
# Each column represents a different candidate function evaluated at all time points
theta = np.column_stack((ones, x, y, z, xy, xz, yz, x2, y2, z2))

# Remove the last row to match dimensions with derivative data
# (derivatives have one fewer point due to finite difference method)
theta = theta[:-1]

def SINDy(theta, dXdt, lambd, n):
    """
    Implementation of the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm
    
    Parameters:
    theta - Library of candidate functions evaluated at each data point
    dXdt - Time derivatives of state variables
    lambd - Sparsification threshold value
    n - Number of state variables
    
    Returns:
    Xi - Sparse coefficient matrix representing the identified system
    """
    # Initial least-squares regression to get a first estimate of coefficients
    Xi = np.linalg.lstsq(theta, dXdt)[0]  # lstsq returns a tuple, extract just the coefficients
    
    # Iterative sparsification process (sequential thresholding least squares)
    for k in range(10):  # Perform 10 iterations of sparsification
        # Identify coefficients below the threshold
        smallinds = np.abs(Xi) < lambd  # Boolean matrix: True where |coefficient| < threshold
        
        # Set small coefficients to zero (enforcing sparsity)
        Xi[smallinds] = 0
        
        # Re-fit the model using only the non-zero coefficients
        for var in range(n):  # Loop through each state variable
            # Find indices of coefficients that survived thresholding
            biginds = smallinds[:,var] == 0
            
            # Re-compute coefficients using only the significant terms
            # This focuses the model on the most important dynamics
            Xi[biginds,var] = np.linalg.lstsq(theta[:,biginds], dXdt[:,var])[0]
    
    return Xi  # Return the sparse coefficient matrix

# Set the sparsification threshold - controls how aggressively to simplify the model
lambd = 0.1

# Apply the SINDy algorithm to identify governing equations from data
Xi = SINDy(theta, dX, lambd, n=3)  # n=3 because we have 3 state variables (x, y, z)

# Print the identified coefficient matrix
# Non-zero entries correspond to terms in the discovered governing equations
# Rows represent different candidate functions
# Columns represent equations for dx/dt, dy/dt, dz/dt respectively
print(Xi)