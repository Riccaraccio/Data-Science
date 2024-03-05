import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.datasets._samples_generator import make_blobs 

X, Y = make_blobs(n_samples=100, centers=2, cluster_std=0.6, random_state=0)

## Dendrograms

distances = pdist(X,metric='euclidean') #calculate pairwise distances
Z = hierarchy.linkage(distances,method='average') #perform agglomeration
# It takes 85% of the maximum value of the third column of the linkage matrix Z, which typically represents the distance between merged clusters.
thresh = 0.85*np.max(Z[:,2]) 

plt.figure()
#The p parameter specifies the number of points at which to truncate the dendrogram
dn = hierarchy.dendrogram(Z,p=100,color_threshold=thresh) #plot dendograms
plt.axis('off')

plt.show()
