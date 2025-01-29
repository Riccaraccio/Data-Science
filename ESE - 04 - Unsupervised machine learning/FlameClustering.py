"""Combustion Flow Field Analysis using K-means Clustering.

This code analyzes a combustion flow field dataset by:
1. Loading and visualizing physical quantities (Temperature, YOH, YH2)
2. Performing feature standardization
3. Applying K-means clustering to identify distinct flow regions for chemistry integration

The dataset contains a 2D grid of combustion flow measurements including
temperature and species concentrations (YOH, YH2). The analysis helps
identify coherent structures in the combustion flow field.

Data Structure
-------------
- Information stored in info.json
- Grid coordinates stored in X_m.dat and Y_m.dat
- Physical quantities stored in separate .dat files

Data from: https://blastnet.github.io/sharma2024
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import os

# Change working directory to the PCA analysis folder
os.chdir("ESE - 03 - PCA")

# Load metadata containing grid dimensions
metadata = json.load(open("dataset/info.json"))
Nx, Ny = metadata["global"]["Nxyz"]

# Import grid coordinates (X and Y positions)
# Data is stored in little-endian 32-bit float format
X = np.fromfile("dataset/grid/X_m.dat", dtype="<f4").reshape(Ny, Nx)
Y = np.fromfile("dataset/grid/Y_m.dat", dtype="<f4").reshape(Ny, Nx)

# Load physical quantities (Temperature, YOH, YH2) and transpose to match grid dimensions
T = np.fromfile("dataset/data/T_K_id0100.dat", dtype="<f4").reshape(Nx, Ny).T
YOH = np.fromfile("dataset/data/YOH_id0100.dat", dtype="<f4").reshape(Nx, Ny).T
YH2 = np.fromfile("dataset/data/YH2_id0100.dat", dtype="<f4").reshape(Nx, Ny).T

# Create visualization of the three physical quantities
fig, ax = plt.subplots(1, 3)
# Plot Temperature
ax[0].set_title("Temperature")
ax[0].pcolormesh(Y, X, T, cmap="inferno")

# Plot YOH concentration
ax[1].set_title("YOH")
ax[1].pcolormesh(Y, X, YOH, cmap="inferno")

# Plot YH2 concentration
ax[2].set_title("YH2")
ax[2].pcolormesh(Y, X, YH2, cmap="inferno")
plt.tight_layout()
plt.show()

# Import all available features from data directory
features = np.array([])
for feature in os.listdir("dataset/data"):
    if feature.endswith(".dat"):  # Process only .dat files
        if features.size == 0:  # For first feature
            features = np.fromfile(f"dataset/data/{feature}", dtype="<f4")
        else:  # Stack subsequent features vertically
            features = np.vstack((features, np.fromfile(f"dataset/data/{feature}", dtype="<f4")))

# Standardize features (zero mean and unit variance)
from sklearn.preprocessing import StandardScaler
features = StandardScaler().fit_transform(features)

# Perform K-means clustering with 10 clusters
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(features.T)  # Transpose features for clustering
colors = kmeans.labels_.reshape(Nx, Ny).T  # Reshape cluster labels to match grid

# Visualize clustering results
plt.title("K-means Clustering")
plt.pcolormesh(Y, X, colors, cmap=plt.get_cmap('viridis', 10))
plt.colorbar()
plt.show()