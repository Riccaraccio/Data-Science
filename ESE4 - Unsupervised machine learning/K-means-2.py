import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets._samples_generator import make_blobs #to generate data clusters
from sklearn.cluster import KMeans

X, Y = make_blobs(n_samples=200, centers=5, cluster_std=0.6, random_state=0)

#kmeans 
kmeans = KMeans(n_clusters=5) #initiate and n cluster declaration
kmeans.fit(X) #fit points
y_kmeans = kmeans.labels_ #labels of each point

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()