import numpy as np 
import matplotlib.pyplot as plt

# Load the frames array from the file
frames_array = np.load('frames_array.npy')
print("Frames array shape:", frames_array.shape)

# animate the frames
plt.ion()
for frame in frames_array:
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.pause(0.01)
    plt.clf()
plt.ioff()

# Flatten the frames array
flattened_frames = frames_array.reshape(frames_array.shape[0], -1).T

# Perform SVD on the flattened frames
U, S, Vt = np.linalg.svd(flattened_frames, full_matrices=False)
print("Shapes of U, S, Vt:", U.shape, S.shape, Vt.shape)

# Number of singular vectors to display
n = 4

# Plot the first n singular vectors in subplots
fig, axs = plt.subplots(1, n, figsize=(16, 4))

for i in range(n):
    # Reshape the i-th left singular vector into the original frame shape
    reconstructed_frame = U[:, i].reshape(frames_array.shape[1], frames_array.shape[2])
    
    # Plot the reconstructed mode
    axs[i].imshow(reconstructed_frame, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f"U[{i}]")

plt.show()

plt.imshow(np.diag(S))
plt.show()

# a(t) 
a = np.diag(S) @ Vt
print(a.shape)

# temporal modes
plt.plot(a[0, :], label='mode 1')
plt.plot(a[1, :], label='mode 2')
plt.plot(a[2, :], label='mode 3')
plt.plot(a[3, :], label='mode 4')
plt.legend()
plt.show()

# variance of each mode

plt.scatter(range(len(S)), S, marker="o", color="black")
plt.xlabel('Mode')
plt.ylabel('Variance')
plt.yscale('log')
plt.show()

# plot the first frame and the reconstructed frame using the first n singular vectors
n = 35

# Reconstruct the frames using the first n singular vectors
reconstructed_frames = (U[:, :n] @ np.diag(S[:n]) @ Vt[:n, :])
print("Reconstructed frames shape:", reconstructed_frames.shape)

# calculate pointwise error
error = (flattened_frames - reconstructed_frames)/flattened_frames

error = error.T.reshape(frames_array.shape)
print("Error shape:", error.shape)
print("Frames array shape:", frames_array.shape)

# Reshape the reconstructed frames array
reconstructed_frames = reconstructed_frames.T.reshape(frames_array.shape)

plt.ion()
# Plot the first frame and the reconstructed frame
fig, axs = plt.subplots(1, 3)

for i in range(reconstructed_frames.shape[0]):

    # Original first frame
    axs[0].imshow(frames_array[i], cmap='gray')
    axs[0].axis('off')
    axs[0].set_title("Original Video")

    # Reconstructed first frame
    axs[1].imshow(reconstructed_frames[i], cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f"Reconstructed video using {n} modes")
    
    axs[2].imshow(error[i], vmax=1, vmin=0)
    axs[2].set_title("Error")
    
    plt.axis('off')
    plt.show()
    plt.pause(0.01)
    
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

plt.ioff()
