import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_olivetti_faces

faces, target = fetch_olivetti_faces(return_X_y=True)
print(faces.shape) #check the dimensions

#look at the first face 
plt.imshow(faces[0].reshape(64,64), cmap='gray')
plt.show()

# # We import the PCA object from the sklearn package
from sklearn.decomposition import PCA

# initialize pca object
pca = PCA() 
pca.fit(faces)

eigenfaces = pca.components_.T # U matrix
singular_values= pca.singular_values_ # S matrix

# Print the first 10 eigenfaces
fig, ax = plt.subplots(2,5)
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(eigenfaces[:,i*5+j].reshape(64,64), cmap='gray')
        ax[i,j].set_title(str(singular_values[i*5+j].round(2)))
        ax[i,j].axis('off')
plt.show()

#reconstruct the first face using the first 10 eigenfaces
n_eigenfaces = 100
reconstructed_face = faces[0] @ eigenfaces[:,:n_eigenfaces] @ eigenfaces[:,:n_eigenfaces].T

fig, ax = plt.subplots(1, 2)
ax[0].imshow(faces[0].reshape(64,64), cmap='gray')
ax[0].set_title('Original Face')
ax[0].axis('off')

ax[1].imshow(reconstructed_face.reshape(64,64), cmap='gray')
ax[1].set_title('Reconstructed Face')
ax[1].axis('off')

plt.show()
