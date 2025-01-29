"""Image Compression using SVD (Singular Value Decomposition).

This code demonstrates image compression using SVD by creating low-rank approximations
of a grayscale image. It shows both visual results at different compression levels
and analyzes the importance of singular values in representing the image.

The compression works by keeping only the r most significant singular values and their
corresponding singular vectors, where r is the truncation rank. Lower r means higher
compression but potentially lower image quality.

Methods
-------
1. Convert RGB image to grayscale
2. Perform SVD decomposition
3. Create compressed versions using different truncation ranks
4. Analyze singular value distribution and cumulative importance
"""

from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt

# Load and convert image to grayscale
image = imread("ESE - 02 - SVD/StillLife.jpg")
image_grayscale = np.mean(image, axis=-1)  # Average RGB channels

# Perform SVD decomposition
U, S, Vt = np.linalg.svd(image_grayscale, full_matrices=False)
S = np.diag(S)  # Convert singular values to diagonal matrix

# Create and display compressed versions
i = 1  # Subplot counter
for r in (5, 20, 100):  # Different compression levels
   compressed_image = U[:,:r] @ S[:r, :r] @ Vt[:r, :]  # Create rank-r approximation
   plt.subplot(1, 4, i)
   plt.imshow(compressed_image, cmap="grey")
   plt.title("r = " + str(r))
   plt.axis("off")
   i += 1

# Display original image for comparison
plt.subplot(1, 4, i)
plt.imshow(image_grayscale, cmap="grey")
plt.title("Original Image")
plt.axis("off")
plt.show()

# Analyze singular values
plt.subplot(1, 2, 1)

plt.title("Singular Values")
plt.semilogy(np.diag(S))  # Plot singular values on log scale

plt.subplot(1, 2, 2)
plt.title("Cumulative Normalized Sum")
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))  # Show cumulative importance
plt.show()