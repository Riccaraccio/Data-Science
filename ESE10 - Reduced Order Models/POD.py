import numpy as np 
import matplotlib.pyplot as plt

# Load the frames array from the file
frames_array = np.load('frames_array_color.npy')

# Convert the frames to grayscale using numpy
frames_array_gray = frames_array.mean(axis=-1).astype(np.uint8)

# Flatten the frames array
flattened_frames = frames_array_gray.reshape(frames_array.shape[0], -1).T

# Perform SVD on the flattened frames
U, S, Vt = np.linalg.svd(flattened_frames, full_matrices=False)
print("Shapes of U, S, Vt:", U.shape, S.shape, Vt.shape)

"""# Number of singular vectors to display
n = 4

# Plot the first n singular vectors in subplots
fig, axs = plt.subplots(1, n, figsize=(16, 4))

for i in range(n):
    # Reshape the i-th left singular vector into the original frame shape
    reconstructed_frame = U[:, i].reshape(frames_array_gray.shape[1], frames_array_gray.shape[2])
    
    # Plot the reconstructed frame
    axs[i].imshow(reconstructed_frame, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f"U[{i}]")

plt.show()"""

"""
plt.imshow(np.diag(S))
plt.show()"""

# a(t) 

a = np.diag(S) @ Vt
print(a.shape)

# temporal modes
plt.plot(a[0, :], label='mode 1')
plt.plot(a[1, :], label='mode 2')
plt.plot(a[2, :], label='mode 3')
plt.legend()
plt.show()

# variance of each mode
variance = np.sum(S**2)
variance_modes = S**2 / variance
plt.scatter(range(len(variance_modes)), variance_modes)
plt.xlabel('Mode')
plt.ylabel('Variance')
plt.yscale('log')
plt.show()
plt.show()