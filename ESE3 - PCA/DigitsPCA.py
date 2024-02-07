import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits_images = load_digits().data  # images dataset size 1767x64
digits_labels = load_digits().target # images labels size 1767

print(digits_images.shape) #check the dimensions

# select the index of the digit to show
index = 0

# select the image and reshape it to show it
digit = digits_images[index,:].reshape(8,8)
plt.imshow(digit, cmap='Greys') #More shades than "grey"
plt.show()

# We import the PCA object from the sklearn package
from sklearn.decomposition import PCA

pca = PCA(n_components=64) # we select all the features for now
pca.fit(digits_images)

# We use the attributes of PCA to find the explained variance in percentage
variance_ratio = pca.explained_variance_ratio_ * 100

fig, axs = plt.subplots(1,2, figsize=(8,4))
axs[0].scatter(np.linspace(1, 64, 64), variance_ratio)
axs[0].set_title('explained variance')
axs[1].plot(np.cumsum(variance_ratio))
axs[1].set_title('cumulative explained variance')
plt.show()

# We calculate the matrix A, which contains the directions of the PCs
A = pca.components_.T

# We can plot the first PC
PCindex = 0

plt.imshow(A[:,PCindex].reshape(8,8), cmap="grey")
plt.title("PC number " + str(PCindex+1))
plt.show()

# We calculate the PCA scores as Z = X A
Z = digits_images @ A

q = 20 # number of principal components to use
index = 0 # the index of the digit that we want to reconstruct 

digit_orig = digits_images[index,:] # original digit
digit_rec = Z[index,:q] @ A[:,:q].T        # reconstructed digit

# We can plot them side by side 
fig, axs = plt.subplots(1,2)
axs[0].imshow(digit_orig.reshape(8,8), cmap='Greys')
axs[0].set_title('original')
axs[1].imshow(digit_rec.reshape(8,8), cmap='Greys')
axs[1].set_title('reconstructed with ' + str(q) + ' PCs')
plt.show()

plt.scatter(Z[:,0], Z[:,1], c=digits_labels, alpha=0.5, cmap="jet") 
# alpha = transparency
# c = to differentiate between digits
plt.colorbar()
plt.show()
