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

# initialize pca object
pca = PCA() # we select all the features for now
pca.fit(digits_images)

# We calculate the matrix U, which contains the directions of the PCs
principal_components = pca.components_.T

# We can plot the first two PCs side by side
fig, axs = plt.subplots(1,2)
axs[0].imshow(principal_components[:,0].reshape(8,8), cmap='Greys')
axs[0].set_title('PC number 1')
axs[1].imshow(principal_components[:,1].reshape(8,8), cmap='Greys')
axs[1].set_title('PC number 2')
plt.show()

# We calculate the PCA scores as Z = X A
pca_scores = digits_images @ principal_components

plt.scatter(pca_scores[:,0], pca_scores[:,1], c=digits_labels, alpha=0.5, s=10, cmap=plt.cm.get_cmap('jet', 10)) 
# alpha = transparency
# c = to differentiate between digits
# cmap=plt.cm.get_cmap('jet', 10) sets the color map to jet and the number of colors to 10
plt.colorbar()
plt.show()
