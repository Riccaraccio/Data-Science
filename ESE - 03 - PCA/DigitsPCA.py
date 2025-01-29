"""Principal Component Analysis (PCA) of Handwritten Digits.

This code demonstrates PCA application on the sklearn digits dataset,
which contains 1,767 8x8 grayscale images of handwritten digits (0-9).
The analysis shows how PCA can be used for dimensionality reduction
and visualization of high-dimensional image data.

The code visualizes:
1. An example digit image
2. The first two principal components
3. A scatter plot of digits projected onto these components
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load the digits dataset
digits_images = load_digits().data  # 1767 images, each 8x8 pixels (64 features)
digits_labels = load_digits().target  # corresponding digit labels (0-9)
print(digits_images.shape)

# Display an example digit
index = 0
digit = digits_images[index,:].reshape(8,8)  # reshape to 8x8 image
plt.imshow(digit, cmap='Greys')
plt.show()

# Perform PCA
pca = PCA()  # initialize PCA object
pca.fit(digits_images)

# Extract principal components
principal_components = pca.components_.T  # each column is a principal component

# Visualize first two principal components
fig, axs = plt.subplots(1,2)
axs[0].imshow(principal_components[:,0].reshape(8,8), cmap='Greys')
axs[0].set_title('PC number 1')
axs[1].imshow(principal_components[:,1].reshape(8,8), cmap='Greys')
axs[1].set_title('PC number 2')
plt.show()

# Project data onto principal components
pca_scores = digits_images @ principal_components

# Create scatter plot of digits in PC space
plt.scatter(pca_scores[:,0], pca_scores[:,1],  # plot first two PCs
          c=digits_labels,  # color by digit
          alpha=0.5,  # transparency
          s=10,  # point size
          cmap=plt.cm.get_cmap('jet', 10))  # colormap with 10 colors
plt.colorbar()
plt.show()