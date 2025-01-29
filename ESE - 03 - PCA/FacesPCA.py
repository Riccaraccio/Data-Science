"""Face Image Analysis and Reconstruction using PCA (Eigenfaces).

This code demonstrates the application of PCA to facial image analysis using
the Olivetti faces dataset. It shows how faces can be decomposed into and
reconstructed from eigenfaces (principal components of face images).

The Olivetti dataset contains 400 64x64 grayscale images of faces.
The code shows:
1. Original face visualization
2. Top eigenfaces extraction
3. Face reconstruction using eigenfaces
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Load the Olivetti faces dataset
faces, target = fetch_olivetti_faces(return_X_y=True)
print(faces.shape)  # Show dimensions (400 faces, 4096 pixels each)

# Display first face
plt.imshow(faces[0].reshape(64,64), cmap='gray')
plt.show()

# Perform PCA to extract eigenfaces
pca = PCA() 
pca.fit(faces)

eigenfaces = pca.components_.T  # Each column is an eigenface
singular_values = pca.singular_values_  # Importance of each eigenface

# Visualize top 10 eigenfaces with their singular values
fig, ax = plt.subplots(2,5)
for i in range(2):
   for j in range(5):
       ax[i,j].imshow(eigenfaces[:,i*5+j].reshape(64,64), cmap='gray')
       ax[i,j].set_title(str(singular_values[i*5+j].round(2)))
       ax[i,j].axis('off')
plt.show()

# Reconstruct a face using first n eigenfaces
n_eigenfaces = 100
reconstructed_face = faces[0] @ eigenfaces[:,:n_eigenfaces] @ eigenfaces[:,:n_eigenfaces].T

# Compare original and reconstructed face
fig, ax = plt.subplots(1, 2)
ax[0].imshow(faces[0].reshape(64,64), cmap='gray')
ax[0].set_title('Original Face')
ax[0].axis('off')

ax[1].imshow(reconstructed_face.reshape(64,64), cmap='gray')
ax[1].set_title('Reconstructed Face')
ax[1].axis('off')
plt.show()