"""Image Color Quantization using K-means Clustering.

This code demonstrates color quantization of an image using K-means clustering,
reducing the image from millions of possible colors to just 16 colors while
preserving the main visual features. The process involves:
1. Converting the image into a collection of RGB points
2. Clustering these points in RGB space
3. Replacing each pixel's color with its cluster center
4. Reconstructing the image with the reduced color palette

The visualization shows both the clustering in RGB space and the
resulting recolored image.
"""

from matplotlib.image import imread
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import MiniBatchKMeans

# Load and reshape image
image = imread("ESE - 04 - Unsupervised machine learning/StillLife.jpg")
data = image.reshape(image.shape[0]*image.shape[1], image.shape[2])  # Flatten to pixel array

def plot_pixels(data, title, colors=None, N=10000):
   if colors is None:
       colors = data
   
   # Sample random subset of pixels
   rng = np.random.RandomState(0)
   i = rng.permutation(data.shape[0])[:N]
   colors = colors[i]/255  # Normalize colors
   
   R, G, B = data[i].T  # Extract RGB components
   
   # Create scatterplots
   fig, ax = plt.subplots(1, 2)
   ax[0].scatter(R, G, c=colors, marker='.')
   ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 255), ylim=(0, 255))

   ax[1].scatter(R, B, c=colors, marker='.')
   ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 255), ylim=(0, 255))

   fig.suptitle(title, size=20)
   plt.show()

# Show original color distribution
plot_pixels(data, title='Input color space: 16 million possible colors')

# Perform color quantization
kmeans = MiniBatchKMeans(16)  # Create 16-color palette
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
new_colors = new_colors.astype(int)

# Show reduced color distribution
plot_pixels(data, colors=new_colors, title="Reduced color space: 16 colors")

# Compare original and recolored images
image_recolored = new_colors.reshape(image.shape)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(image_recolored)
ax[1].set_title('16-color Image')
plt.show()