import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=0) #load the Olivetti faces dataset
print(faces.shape)

n_samples, n_features = faces.shape
image_shape = (64, 64) # the images are 64x64 pixels

plt.imshow(faces[1].reshape(image_shape),cmap="grey") #print the 1st face
plt.show()

#CORRELATION MATRIX
A = faces[:5,:] #three face from 2 differnet person, 64 each
print(A.shape)
CorrMatrix =  A.T @ A

plt.imshow(CorrMatrix,cmap="grey")
plt.colorbar()
plt.show()

# REF FACE
plt.subplot(1,3,1)
plt.imshow(A[1,:].reshape(image_shape).T,cmap="grey")

#LOOK ALIKE FACE
plt.subplot(1,3,2)
plt.imshow(np.reshape(A[:,2],(168,192)).T,cmap="grey")

#DIFFERENT FACE
plt.subplot(1,3,3)
plt.imshow(np.reshape(A[:,3],(168,192)).T,cmap="grey")

plt.show()