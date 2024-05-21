import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = np.load('RosslerAttractor.npy')
print("Data shape:", X.shape)

# Compute the time derivatives
dt = 0.01
dX = (X[1:] - X[:-1]) / dt

# Construct the library
# We will use a library of monomials up to degree 2

Z = X[:-1] # cut off the last element to match the size of dX

Z = np.vstack((Z.T, (Z[:,0]*Z[:,1]).T, (Z[:,0]*Z[:,2]).T, (Z[:,1]*Z[:,2]).T, (Z[:,0]**2).T, (Z[:,1]**2).T, (Z[:,2]**2).T)).T
