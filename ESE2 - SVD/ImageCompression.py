
from matplotlib.image import imread #to read the image
import numpy as np #numpy
import matplotlib.pyplot as plt #to show the image

A = imread("StillLife.jpg")
X = np.mean(A, axis=-1) #averaging over the last column, the RGB scale

#TAKE THE SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False) 
S = np.diag(S) #SVD returns the vector on the diagonal, we need to reconstruct the matrix

i = 1 #figure index

for r in (5 , 20 , 100): #define the truncation rank r
    XApprox = U[:,:r] @ S[:r, :r] @ Vt[:r, :] #@ operator performs matrix multiplication
    plt.subplot(1, 4, i) #create subplot 
    plt.imshow(XApprox, cmap="grey") # plot current compressed image
    plt.title("r = " + str(r))
    plt.axis("off")
    i += 1 #increase subplot index

plt.subplot(1, 4 ,i)
plt.imshow(X, cmap="grey") #plot original image
plt.title("Original Image")
plt.axis("off")
plt.show()

# SINGULAR VALUE 
plt.subplot(1,2,1)
plt.title("Singular Values")
plt.semilogy(np.diag(S)) #S contains the singular value ordered in descending order

plt.subplot(1,2,2)
plt.title("Cumulative Normalized Sum") #shows the importace of the singular values
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.show()