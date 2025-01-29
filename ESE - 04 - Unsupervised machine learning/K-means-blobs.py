"""K-means Clustering Visualization on Synthetic Data.

This code demonstrates K-means clustering on synthetic 2D data by:
1. Generating 5 Gaussian clusters using make_blobs
2. Applying K-means to identify cluster centers and assignments
3. Visualizing the results with color-coded points and cluster centers

The synthetic dataset consists of 200 points distributed around 
5 centers with a standard deviation of 0.6.
"""

import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic clustered data
X, Y = make_blobs(n_samples=200,    # Total number of points
                 centers=5,         # Number of clusters
                 cluster_std=0.6,   # Cluster spread
                 random_state=0)    # For reproducibility

# Perform K-means clustering
kmeans = KMeans(n_clusters=5)  # Initialize with 5 clusters
kmeans.fit(X)                  # Fit model to data
y_kmeans = kmeans.labels_      # Get cluster assignments

# Visualize results
plt.scatter(X[:, 0], X[:, 1],     # Plot data points
          c=y_kmeans,             # Color by cluster assignment
          s=50,                   # Point size
          cmap='viridis')         # Color scheme

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],  # Plot centers
          c='black',                      # Center color
          s=200,                          # Center point size
          alpha=0.5)                      # Transparency
plt.show()