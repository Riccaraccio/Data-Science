import numpy as np
import scipy # to load the matrix
import matplotlib.pyplot as plt

mat_content = scipy.io.loadmat("allFaces.mat") #load matrix file
X = mat_content["faces"] #data is in colum "faces" of mat_content
print(X.shape)

plt.imshow(np.reshape(X[:,0],(168,192)).T,cmap="grey") #print the 1st face
plt.show()

#CORRELATION MATRIX
A = np.concatenate((X[:,0:3], X[:, 64:67]),axis=1) #three face from 2 differnet person, 64 each

CorrMatrix =  A.T @ A

plt.imshow(CorrMatrix,cmap="grey")
plt.colorbar()
plt.show()

# REF FACE
plt.subplot(1,3,1)
plt.imshow(np.reshape(A[:,1],(168,192)).T,cmap="grey")

#LOOK ALIKE FACE
plt.subplot(1,3,2)
plt.imshow(np.reshape(A[:,2],(168,192)).T,cmap="grey")

#DIFFERENT FACE
plt.subplot(1,3,3)
plt.imshow(np.reshape(A[:,3],(168,192)).T,cmap="grey")

plt.show()