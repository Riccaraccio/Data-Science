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

# Sample random s# Load image and prepare data
image = imread("ESE-04-Unsupervised machine learning/StillLife.jpg")
pixels = image.reshape(-1, image.shape[2])  # Flatten to pixel array

def plot_colors(data, title, colors=None, samples=10000):
    # Sample random pixels for visualization
    if colors is None:
        colors = data
    
    rng = np.random.RandomState(0)
    indices = rng.permutation(len(data))[:samples]
    
    R, G, B = data[indices].T  # Extract RGB components
    norm_colors = colors[indices]/255  # Normalize colors
    
    # Create side-by-side scatterplots
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(R, G, c=norm_colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 255), ylim=(0, 255))

    ax[1].scatter(R, B, c=norm_colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 255), ylim=(0, 255))

    fig.suptitle(title, size=20)
    plt.show()

# Show original colors
plot_colors(pixels, title='Original: 16 million possible colors')

# Perform color quantization with K-means
kmeans = MiniBatchKMeans(n_clusters=16)
kmeans.fit(pixels)
reduced_colors = kmeans.cluster_centers_[kmeans.predict(pixels)].astype(int)

# Show reduced color distribution
plot_colors(pixels, colors=reduced_colors, title="Reduced: 16 colors")

# Compare original and recolored images
recolored = reduced_colors.reshape(image.shape)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(recolored)
ax[1].set_title('16-color Image')
plt.show()